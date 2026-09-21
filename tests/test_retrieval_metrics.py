import pytest

from app.evaluation.retrieval_metrics import (
    EvaluationQuery,
    mean_reciprocal_rank,
    reciprocal_rank,
    recall_at_k,
)


def test_recall_at_k_returns_one_when_relevant_chunk_is_found():
    score = recall_at_k(
        retrieved_chunk_indexes=[3, 1, 5],
        relevant_chunk_indexes={1},
        k=3,
    )

    assert score == 1.0


def test_recall_at_k_returns_zero_when_relevant_chunk_is_missing():
    score = recall_at_k(
        retrieved_chunk_indexes=[3, 5, 7],
        relevant_chunk_indexes={1},
        k=3,
    )

    assert score == 0.0


def test_recall_at_k_handles_multiple_relevant_chunks():
    score = recall_at_k(
        retrieved_chunk_indexes=[1, 4, 7, 9],
        relevant_chunk_indexes={1, 7},
        k=3,
    )

    assert score == 1.0


def test_reciprocal_rank_returns_one_for_first_result():
    score = reciprocal_rank(
        retrieved_chunk_indexes=[4, 2, 8],
        relevant_chunk_indexes={4},
    )

    assert score == 1.0


def test_reciprocal_rank_returns_half_for_second_result():
    score = reciprocal_rank(
        retrieved_chunk_indexes=[4, 2, 8],
        relevant_chunk_indexes={2},
    )

    assert score == 0.5


def test_reciprocal_rank_returns_zero_when_not_found():
    score = reciprocal_rank(
        retrieved_chunk_indexes=[4, 2, 8],
        relevant_chunk_indexes={9},
    )

    assert score == 0.0


def test_mean_reciprocal_rank_calculates_average():
    queries = [
        EvaluationQuery(
            query="health insurance",
            relevant_chunk_indexes={1},
        ),
        EvaluationQuery(
            query="vacation policy",
            relevant_chunk_indexes={2},
        ),
    ]

    retrieved_results = [
        [1, 5, 7],
        [5, 2, 8],
    ]

    score = mean_reciprocal_rank(
        queries,
        retrieved_results,
    )

    assert score == 0.75


def test_mean_reciprocal_rank_rejects_mismatched_lengths():
    queries = [
        EvaluationQuery(
            query="health insurance",
            relevant_chunk_indexes={1},
        ),
    ]

    with pytest.raises(ValueError, match="same length"):
        mean_reciprocal_rank(
            queries,
            [],
        )


def test_recall_at_k_rejects_invalid_k():
    with pytest.raises(ValueError, match="k"):
        recall_at_k(
            retrieved_chunk_indexes=[1, 2],
            relevant_chunk_indexes={1},
            k=0,
        )
