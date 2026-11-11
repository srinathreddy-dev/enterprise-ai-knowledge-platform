from unittest.mock import Mock
from types import SimpleNamespace

from fastapi.testclient import TestClient

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from app.generation.answer_service import AnswerGenerationService
from app.generation.rag_factory import build_rag_service
from app.generation.rag_service import RAGResponse
from app.ingestion.chunker import TextChunk
from app.main import app, set_rag_service
from app.reranking.cross_encoder_reranker import RerankedResult
from app.retrieval.bm25_retriever import RetrievalResult


client = TestClient(
    app,
    headers={"X-API-Key": "test-api-key"},
)


def create_mock_openai_response(status_code: int):
    """Create a mock HTTP response for OpenAI API exceptions."""

    response = Mock()
    response.status_code = status_code
    response.headers = {}
    response.request = Mock()

    response.json.return_value = {
        "error": {"message": "Mock OpenAI API error"}
    }

    return response


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ask_endpoint_rejects_missing_api_key():
    unauthenticated_client = TestClient(app)

    response = unauthenticated_client.post(
        "/ask",
        json={"question": "What is the vacation policy?"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key."

def test_ask_endpoint_rejects_invalid_api_key():
    response = client.post(
        "/ask",
        json={"question": "What is the vacation policy?"},
        headers={"X-API-Key": "incorrect-api-key"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key."


def test_ask_endpoint_returns_503_when_api_key_not_configured(
    monkeypatch,
):
    monkeypatch.setattr(
        "app.main.settings",
        SimpleNamespace(app_api_key=None),
    )

    response = client.post(
        "/ask",
        json={"question": "What is the vacation policy?"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "API authentication is not configured."
    )


def test_ask_endpoint_returns_answer_and_evidence():
    chunk = TextChunk(
        text="Employees receive 15 days of paid vacation per year.",
        page_number=1,
        chunk_index=0,
        document_id="employee_handbook",
    )

    evidence = [
        RerankedResult(
            result=RetrievalResult(chunk=chunk, score=0.9),
            score=8.5,
        )
    ]

    mock_service = Mock()

    mock_service.ask.return_value = RAGResponse(
        question="How many vacation days do employees receive?",
        answer=(
            "Employees receive 15 days of paid vacation per year. "
            "[Page 1]"
        ),
        evidence=evidence,
    )

    set_rag_service(mock_service)

    try:
        response = client.post(
            "/ask",
            json={
                "question": "How many vacation days do employees receive?"
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["question"] == (
            "How many vacation days do employees receive?"
        )

        assert data["answer"] == (
            "Employees receive 15 days of paid vacation per year. "
            "[Page 1]"
        )

        assert len(data["evidence"]) == 1

        assert data["evidence"][0]["chunk_id"] == (
            "employee_handbook-page-1-chunk-0"
        )

        assert data["evidence"][0]["document_name"] == (
            "employee_handbook.pdf"
        )

    finally:
        set_rag_service(None)


def test_ask_endpoint_returns_503_when_service_is_not_configured():
    set_rag_service(None)

    response = client.post(
        "/ask",
        json={"question": "What is the vacation policy?"},
    )

    assert response.status_code == 503


def test_ask_endpoint_rejects_empty_question():
    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_ask_endpoint_handles_missing_evidence():
    mock_service = Mock()

    mock_service.ask.return_value = RAGResponse(
        question="What is the vacation policy?",
        answer="I could not find relevant information.",
        evidence=[],
    )

    set_rag_service(mock_service)

    try:
        response = client.post(
            "/ask",
            json={"question": "What is the vacation policy?"},
        )

        assert response.status_code == 200
        assert response.json()["evidence"] == []

    finally:
        set_rag_service(None)


def test_ask_endpoint_with_real_pdf_and_fake_answer_service():
    answer_service = Mock(spec=AnswerGenerationService)

    answer_service.generate_answer.return_value = (
        "Employees receive 15 days of paid vacation per year. "
        "[Page 1]"
    )

    service = build_rag_service(
        pdf_path="data/sample/employee_handbook.pdf",
        answer_service=answer_service,
    )

    set_rag_service(service)

    try:
        response = client.post(
            "/ask",
            json={
                "question": "How many vacation days do employees receive?"
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert "15 days" in data["answer"]
        assert data["evidence"]

        assert data["evidence"][0]["chunk_id"] == (
            "employee_handbook-page-1-chunk-0"
        )

        assert data["evidence"][0]["document_name"] == (
            "employee_handbook.pdf"
        )

        answer_service.generate_answer.assert_called_once()

    finally:
        set_rag_service(None)


def test_ask_endpoint_searches_multiple_pdfs():
    answer_service = Mock(spec=AnswerGenerationService)

    answer_service.generate_answer.return_value = (
        "Multi-factor authentication is required. [Page 1]"
    )

    service = build_rag_service(
        documents_directory="data/sample",
        answer_service=answer_service,
    )

    set_rag_service(service)

    try:
        response = client.post(
            "/ask",
            json={
                "question": "Is multi-factor authentication required?"
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["answer"] == (
            "Multi-factor authentication is required. [Page 1]"
        )

        assert data["evidence"]

        document_names = {
            item["document_name"]
            for item in data["evidence"]
        }

        assert "it_security_policy.pdf" in document_names

        for item in data["evidence"]:
            assert item["document_name"].endswith(".pdf")
            assert item["page_number"] >= 1
            assert item["chunk_id"]

        answer_service.generate_answer.assert_called_once()

    finally:
        set_rag_service(None)


def test_ready_endpoint_returns_ready_when_service_is_configured():
    mock_service = Mock()

    set_rag_service(mock_service)

    try:
        response = client.get("/ready")

        assert response.status_code == 200
        assert response.json() == {"status": "ready"}

    finally:
        set_rag_service(None)


def test_ready_endpoint_returns_503_when_service_is_not_configured():
    set_rag_service(None)

    response = client.get("/ready")

    assert response.status_code == 503

    assert response.json() == {
        "detail": "The knowledge service is not configured."
    }


def test_ask_endpoint_handles_openai_rate_limit():
    mock_service = Mock()

    error = RateLimitError(
        message="Rate limit or quota exceeded",
        response=create_mock_openai_response(429),
        body={"error": {"message": "Quota exceeded"}},
    )

    mock_service.ask.side_effect = error

    set_rag_service(mock_service)

    try:
        response = client.post(
            "/ask",
            json={
                "question": "How many vacation days do employees receive?"
            },
        )

        assert response.status_code == 503

        assert response.json()["detail"] == (
            "The AI service is temporarily unavailable. "
            "Please try again later."
        )

    finally:
        set_rag_service(None)


def test_ask_endpoint_handles_openai_timeout():
    mock_service = Mock()

    error = APITimeoutError(request=Mock())
    mock_service.ask.side_effect = error

    set_rag_service(mock_service)

    try:
        response = client.post(
            "/ask",
            json={
                "question": "How many vacation days do employees receive?"
            },
        )

        assert response.status_code == 504

        assert response.json()["detail"] == (
            "The AI service request timed out. Please try again."
        )

    finally:
        set_rag_service(None)


def test_ask_endpoint_handles_openai_connection_error():
    mock_service = Mock()

    error = APIConnectionError(request=Mock())
    mock_service.ask.side_effect = error

    set_rag_service(mock_service)

    try:
        response = client.post(
            "/ask",
            json={
                "question": "How many vacation days do employees receive?"
            },
        )

        assert response.status_code == 503

        assert response.json()["detail"] == (
            "The AI service is temporarily unreachable."
        )

    finally:
        set_rag_service(None)


def test_ask_endpoint_handles_openai_authentication_error():
    mock_service = Mock()

    error = AuthenticationError(
        message="Invalid API key",
        response=create_mock_openai_response(401),
        body={"error": {"message": "Invalid API key"}},
    )

    mock_service.ask.side_effect = error

    set_rag_service(mock_service)

    try:
        response = client.post(
            "/ask",
            json={
                "question": "How many vacation days do employees receive?"
            },
        )

        assert response.status_code == 502

        assert response.json()["detail"] == (
            "The AI service authentication is not configured correctly."
        )

    finally:
        set_rag_service(None)
