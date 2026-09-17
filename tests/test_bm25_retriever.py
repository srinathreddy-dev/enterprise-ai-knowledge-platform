import pytest

from app.ingestion.chunker import TextChunk
from app.retrieval.bm25_retriever import BM25Retriever


def create_chunks() -> list[TextChunk]:
    """Create representative chunks for retrieval tests."""
    return [
        TextChunk(
            text="Employees can enroll in health insurance during open enrollment.",
            page_number=1,
            chunk_index=0,
        ),
        TextChunk(
            text="Employees receive fifteen paid vacation days each year.",
            page_number=2,
            chunk_index=1,
        ),
        TextChunk(
            text="The company provides retirement savings plans to eligible employees.",
            page_number=3,
            chunk_index=2,
        ),
    ]


def test_bm25_returns_relevant_results():
    retriever = BM25Retriever(create_chunks())

    results = retriever.search(
        "health insurance enrollment",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.page_number == 1
    assert results[0].chunk.chunk_index == 0


def test_bm25_returns_results_in_score_order():
    retriever = BM25Retriever(create_chunks())

    results = retriever.search(
        "employees",
        top_k=3,
    )

    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)


def test_bm25_returns_empty_list_for_empty_query():
    retriever = BM25Retriever(create_chunks())

    results = retriever.search("   ")

    assert results == []


def test_bm25_rejects_empty_chunks():
    with pytest.raises(ValueError, match="At least one chunk"):
        BM25Retriever([])


def test_bm25_rejects_invalid_top_k():
    retriever = BM25Retriever(create_chunks())

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("health insurance", top_k=0)
