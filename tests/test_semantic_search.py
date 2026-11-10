
import numpy as np
import pytest

from app.ingestion.chunker import TextChunk
from app.retrieval.semantic_search import SemanticSearcher


class MockEmbedder:
    def embed_chunks(self, chunks):
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def embed_query(self, query):
        if "health" in query.lower() or "medical" in query.lower():
            return np.array([1.0, 0.0, 0.0])

        if "vacation" in query.lower():
            return np.array([0.0, 1.0, 0.0])

        return np.array([0.0, 0.0, 1.0])


def create_chunks():
    return [
        TextChunk(
            text="Employees can enroll in health insurance.",
            page_number=1,
            chunk_index=0,
        ),
        TextChunk(
            text="Employees receive fifteen paid vacation days.",
            page_number=2,
            chunk_index=1,
        ),
        TextChunk(
            text="The company provides retirement savings plans.",
            page_number=3,
            chunk_index=2,
        ),
    ]


def create_searcher():
    return SemanticSearcher(
        create_chunks(),
        embedder=MockEmbedder(),
    )


def test_semantic_search_returns_relevant_result():
    searcher = create_searcher()

    results = searcher.search(
        "How do employees enroll in medical coverage?",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.page_number == 1


def test_semantic_search_returns_results_in_score_order():
    searcher = create_searcher()

    results = searcher.search("employee vacation time", top_k=3)

    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


def test_semantic_search_returns_empty_list_for_empty_query():
    searcher = create_searcher()

    assert searcher.search("   ") == []


def test_semantic_search_rejects_invalid_top_k():
    searcher = create_searcher()

    with pytest.raises(ValueError, match="top_k"):
        searcher.search("health insurance", top_k=0)


def test_semantic_search_scores_are_cosine_similarity_values():
    searcher = create_searcher()

    results = searcher.search("health insurance", top_k=3)

    for result in results:
        assert -1.0 <= result.score <= 1.0
        assert np.isfinite(result.score)
