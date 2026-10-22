from pathlib import Path

from app.generation.rag_factory import build_rag_service

from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.main import app, set_rag_service
from app.generation.rag_service import RAGResponse
from app.ingestion.chunker import TextChunk
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
        answer="Employees receive 15 days of paid vacation per year. [Page 1]",
        evidence=evidence,
    )
    set_rag_service(mock_service)

    try:
        response = client.post(
            "/ask",
            json={"question": "How many vacation days do employees receive?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == (
            "Employees receive 15 days of paid vacation per year. [Page 1]"
        )
        assert data["evidence"][0]["page_number"] == 1
        assert data["evidence"][0]["chunk_id"] == "page-1-chunk-0"
        assert data["evidence"][0]["relevance_score"] == 8.5
        mock_service.ask.assert_called_once_with(
            "How many vacation days do employees receive?"
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
    assert response.json()["detail"] == (
        "The knowledge service is not configured."
    )


def test_ask_endpoint_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422


def test_ask_endpoint_handles_missing_evidence():
    mock_service = Mock()
    mock_service.ask.side_effect = ValueError(
        "No relevant evidence was retrieved"
    )
    set_rag_service(mock_service)

    try:
        response = client.post(
            "/ask",
            json={"question": "What is the vacation policy?"},
        )

        assert response.status_code == 422
        assert response.json()["detail"] == (
            "No relevant evidence was retrieved"
        )
    finally:
        set_rag_service(None)

def test_ask_endpoint_with_real_sample_pdf():
    pdf_path = Path("data/sample/employee_handbook.pdf")

    class FakeAnswerService:
        def generate_answer(self, question, evidence):
            page = evidence[0].result.chunk.page_number
            return (
                f"Test answer based on page {page}. "
                f"[Page {page}]"
            )

    service = build_rag_service(
        pdf_path=pdf_path,
        answer_service=FakeAnswerService(),
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
        assert data["question"] == (
            "How many vacation days do employees receive?"
        )
        assert data["evidence"]
        assert data["evidence"][0]["page_number"] == 1
        assert "15 days of paid vacation" in (
            data["evidence"][0]["text"]
        )
        assert data["answer"] == (
            "Test answer based on page 1. [Page 1]"
        )
    finally:
        set_rag_service(None)
