
from unittest.mock import Mock

import pytest

from app.generation.rag_service import RAGService
from app.ingestion.chunker import TextChunk
from app.reranking.cross_encoder_reranker import RerankedResult
from app.retrieval.bm25_retriever import RetrievalResult
from app.retrieval.hybrid_retriever import HybridRetrievalResult


@pytest.fixture
def chunks() -> list[TextChunk]:
    return [
        TextChunk(
            text="Employees receive 15 days of paid vacation per year.",
            page_number=1,
            chunk_index=0,
        ),
        TextChunk(
            text="Employees can enroll in the retirement savings plan.",
            page_number=2,
            chunk_index=0,
        ),
    ]


@pytest.fixture
def service(chunks: list[TextChunk]) -> tuple[RAGService, Mock, Mock, Mock]:
    hybrid = Mock()
    reranker = Mock()
    answer_service = Mock()

    hybrid.search.return_value = [
        HybridRetrievalResult(
            chunk=chunks[0],
            score=0.9,
            bm25_score=4.0,
            semantic_score=0.8,
        ),
        HybridRetrievalResult(
            chunk=chunks[1],
            score=0.7,
            bm25_score=2.0,
            semantic_score=0.6,
        ),
    ]

    reranker.rerank.return_value = [
        RerankedResult(
            result=RetrievalResult(chunk=chunks[0], score=0.9),
            score=8.5,
        )
    ]

    answer_service.generate_answer.return_value = (
        "Employees receive 15 days of paid vacation per year. [Page 1]"
    )

    rag_service = RAGService(
        chunks=chunks,
        hybrid_retriever=hybrid,
        reranker=reranker,
        answer_service=answer_service,
        retrieval_top_k=10,
        evidence_top_k=5,
    )

    return rag_service, hybrid, reranker, answer_service


def test_ask_returns_answer_and_evidence(service: tuple) -> None:
    rag_service, _, _, _ = service

    response = rag_service.ask("How many vacation days do employees receive?")

    assert response.answer == (
        "Employees receive 15 days of paid vacation per year. [Page 1]"
    )
    assert response.question == "How many vacation days do employees receive?"
    assert len(response.evidence) == 1
    assert response.evidence[0].result.chunk.page_number == 1


def test_ask_calls_components_in_pipeline_order(service: tuple) -> None:
    rag_service, hybrid, reranker, answer_service = service

    rag_service.ask("How many vacation days do employees receive?")

    hybrid.search.assert_called_once()
    reranker.rerank.assert_called_once()
    answer_service.generate_answer.assert_called_once()

    retrieved = hybrid.search.return_value
    reranker.rerank.assert_called_once_with(
        query="How many vacation days do employees receive?",
        results=[
            RetrievalResult(chunk=item.chunk, score=item.score)
            for item in retrieved
        ],
        top_k=5,
    )


def test_ask_rejects_empty_question(service: tuple) -> None:
    rag_service, hybrid, reranker, answer_service = service

    with pytest.raises(ValueError, match="Question cannot be empty"):
        rag_service.ask("   ")

    hybrid.search.assert_not_called()
    reranker.rerank.assert_not_called()
    answer_service.generate_answer.assert_not_called()


def test_ask_rejects_no_retrieved_evidence(
    chunks: list[TextChunk],
) -> None:
    hybrid = Mock()
    hybrid.search.return_value = []

    rag_service = RAGService(
        chunks=chunks,
        hybrid_retriever=hybrid,
        reranker=Mock(),
        answer_service=Mock(),
    )

    with pytest.raises(ValueError, match="No relevant evidence was retrieved"):
        rag_service.ask("What is the company vacation policy?")

