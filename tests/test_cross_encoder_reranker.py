
import pytest

from app.ingestion.chunker import TextChunk
from app.retrieval.bm25_retriever import RetrievalResult
from app.reranking.cross_encoder_reranker import CrossEncoderReranker


def create_results():
    chunks = [
        TextChunk(
            text="Employees receive annual vacation benefits.",
            page_number=1,
            chunk_index=0,
            document_id="employee_handbook",
        ),
        TextChunk(
            text="Employees enroll in health insurance during open enrollment.",
            page_number=2,
            chunk_index=1,
            document_id="employee_handbook",
        ),
        TextChunk(
            text="Employees can request remote work arrangements.",
            page_number=3,
            chunk_index=2,
            document_id="employee_handbook",
        ),
    ]

    return [
        RetrievalResult(chunk=chunk, score=score)
        for chunk, score in zip(chunks, [0.2, 0.5, 0.3])
    ]


def test_reranker_sorts_results_by_relevance():
    reranker = CrossEncoderReranker()

    results = reranker.rerank(
        query="employees health insurance enrollment",
        results=create_results(),
        top_k=3,
    )

    assert len(results) == 3
    assert results[0].result.chunk.page_number == 2


def test_reranker_returns_requested_top_k():
    reranker = CrossEncoderReranker()

    results = reranker.rerank(
        query="employee benefits",
        results=create_results(),
        top_k=2,
    )

    assert len(results) == 2


def test_reranker_returns_empty_for_empty_query():
    reranker = CrossEncoderReranker()

    assert reranker.rerank(
        query="   ",
        results=create_results(),
    ) == []


def test_reranker_returns_empty_for_no_results():
    reranker = CrossEncoderReranker()

    assert reranker.rerank(
        query="employee benefits",
        results=[],
    ) == []


def test_reranker_rejects_invalid_top_k():
    reranker = CrossEncoderReranker()

    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank(
            query="employee benefits",
            results=create_results(),
            top_k=0,
        )


def test_reranker_preserves_original_chunks():
    reranker = CrossEncoderReranker()
    original = create_results()

    results = reranker.rerank(
        query="employee benefits",
        results=original,
        top_k=3,
    )

    assert {
        item.result.chunk.chunk_id for item in results
    } == {
        item.chunk.chunk_id for item in original
    }
