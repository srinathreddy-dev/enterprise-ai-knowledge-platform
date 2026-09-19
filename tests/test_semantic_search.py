import numpy as np
import pytest

from app.ingestion.chunker import TextChunk
from app.retrieval.semantic_search import SemanticSearcher


def create_chunks() -> list[TextChunk]:
    """Create representative chunks for semantic search tests."""
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


def test_semantic_search_returns_relevant_result():
    searcher = SemanticSearcher(create_chunks())

    results = searcher.search(
        "How do employees enroll in medical coverage?",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.page_number == 1
    assert results[0].chunk.chunk_index == 0


def test_semantic_search_returns_results_in_score_order():
    searcher = SemanticSearcher(create_chunks())

    results = searcher.search(
        "employee vacation time",
        top_k=3,
    )

    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)


def test_semantic_search_returns_empty_list_for_empty_query():
    searcher = SemanticSearcher(create_chunks())

    results = searcher.search("   ")

    assert results == []


def test_semantic_search_rejects_invalid_top_k():
    searcher = SemanticSearcher(create_chunks())

    with pytest.raises(ValueError, match="top_k"):
        searcher.search("health insurance", top_k=0)


def test_semantic_search_scores_are_cosine_similarity_values():
    searcher = SemanticSearcher(create_chunks())

    results = searcher.search(
        "health insurance",
        top_k=3,
    )

    for result in results:
        assert -1.0 <= result.score <= 1.0
        assert np.isfinite(result.score)
