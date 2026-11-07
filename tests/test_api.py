
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.generation.answer_service import AnswerGenerationService
from app.generation.rag_factory import build_rag_service
from app.generation.rag_service import RAGResponse
from app.ingestion.chunker import TextChunk
from app.main import app, set_rag_service
from app.reranking.cross_encoder_reranker import RerankedResult
from app.retrieval.bm25_retriever import RetrievalResult


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ask_endpoint_returns_answer_and_evidence():
    chunk = TextChunk(
        text="Employees receive 15 days of paid vacation per year.",
        page_number=1,
        chunk_index=0,
        document_id="employee_handbook",
    )

    evidence = [
        RerankedResult(
            result=RetrievalResult(
                chunk=chunk,
                score=0.9,
            ),
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
                "question": (
                    "How many vacation days do employees receive?"
                )
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
                "question": (
                    "How many vacation days do employees receive?"
                )
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert "15 days" in data["answer"]

        assert data["evidence"]

        assert data["evidence"][0]["chunk_id"] == (
            "employee_handbook-page-1-chunk-0"
        )

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

