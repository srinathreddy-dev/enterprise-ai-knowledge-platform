
import numpy as np
import pytest

from app.ingestion.chunker import TextChunk
from app.retrieval.cross_encoder_reranker import (
    CrossEncoderReranker,
)


class MockCrossEncoder:
    def predict(self, pairs):
        return np.array([0.2, 0.9, 0.5])


def create_chunks():
    return [
        TextChunk(
            text="Employees receive annual vacation benefits.",
            page_number=1,
            chunk_index=0,
            document_id="employee_handbook",
        ),
        TextChunk(
            text="Employees enroll in health insurance during open enrollment.",
            page_number=2,
            chunk_index=0,
            document_id="employee_handbook",
        ),
        TextChunk(
            text="Employees can request remote work arrangements.",
            page_number=3,
            chunk_index=0,
            document_id="employee_handbook",
        ),
    ]


def create_reranker():
    reranker = CrossEncoderReranker.__new__(
        CrossEncoderReranker
    )
    reranker.model = MockCrossEncoder()
    return reranker


def test_reranker_sorts_chunks_by_relevance():
    reranker = create_reranker()

    results = reranker.rerank(
        query="How do employees enroll in health insurance?",
        chunks=create_chunks(),
        top_k=3,
    )

    assert results[0].score == 0.9
    assert results[1].score == 0.5
    assert results[2].score == 0.2


def test_reranker_returns_requested_top_k():
    reranker = create_reranker()

    results = reranker.rerank(
        query="employee benefits",
        chunks=create_chunks(),
        top_k=2,
    )

    assert len(results) == 2


def test_reranker_returns_empty_for_empty_query():
    reranker = create_reranker()

    results = reranker.rerank(
        query="   ",
        chunks=create_chunks(),
    )

    assert results == []


def test_reranker_returns_empty_for_no_chunks():
    reranker = create_reranker()

    results = reranker.rerank(
        query="employee benefits",
        chunks=[],
    )

    assert results == []


def test_reranker_rejects_invalid_top_k():
    reranker = create_reranker()

    with pytest.raises(ValueError):
        reranker.rerank(
            query="employee benefits",
            chunks=create_chunks(),
            top_k=0,
        )


def test_reranker_results_contain_original_chunks():
    reranker = create_reranker()
    chunks = create_chunks()

    results = reranker.rerank(
        query="employee benefits",
        chunks=chunks,
        top_k=3,
    )

    assert {result.chunk.chunk_id for result in results} == {
        chunk.chunk_id for chunk in chunks
    }
