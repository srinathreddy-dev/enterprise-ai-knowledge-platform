import pytest

from app.ingestion.chunker import TextChunk
from app.retrieval.hybrid_retriever import HybridRetriever


def create_chunks() -> list[TextChunk]:
    """Create representative chunks for hybrid retrieval tests."""
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


def test_hybrid_retriever_returns_relevant_result():
    retriever = HybridRetriever(create_chunks())

    results = retriever.search(
        "How do employees enroll in medical coverage?",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.page_number == 1
    assert results[0].chunk.chunk_index == 0


def test_hybrid_results_are_sorted_by_combined_score():
    retriever = HybridRetriever(create_chunks())

    results = retriever.search(
        "employee vacation time",
        top_k=3,
    )

    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)


def test_hybrid_result_contains_both_retrieval_scores():
    retriever = HybridRetriever(create_chunks())

    results = retriever.search(
        "health insurance",
        top_k=1,
    )

    result = results[0]

    assert isinstance(result.bm25_score, float)
    assert isinstance(result.semantic_score, float)
    assert isinstance(result.score, float)


def test_hybrid_retriever_returns_empty_list_for_empty_query():
    retriever = HybridRetriever(create_chunks())

    results = retriever.search("   ")

    assert results == []


def test_hybrid_retriever_rejects_invalid_top_k():
    retriever = HybridRetriever(create_chunks())

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("health insurance", top_k=0)


def test_hybrid_retriever_rejects_negative_weights():
    retriever = HybridRetriever(create_chunks())

    with pytest.raises(ValueError, match="weights"):
        retriever.search(
            "health insurance",
            bm25_weight=-0.1,
        )


def test_hybrid_retriever_requires_positive_total_weight():
    retriever = HybridRetriever(create_chunks())

    with pytest.raises(ValueError, match="at least one"):
        retriever.search(
            "health insurance",
            bm25_weight=0.0,
            semantic_weight=0.0,
        )
