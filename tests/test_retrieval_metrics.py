import pytest

from app.evaluation.retrieval_metrics import (
    EvaluationQuery,
    mean_reciprocal_rank,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)


def test_recall_at_k_returns_one_when_relevant_chunk_is_found():
    score = recall_at_k(
        retrieved_chunk_ids=[
            "page-1-chunk-0",
            "page-2-chunk-0",
            "page-3-chunk-0",
        ],
        relevant_chunk_ids={"page-2-chunk-0"},
        k=3,
    )

    assert score == 1.0


def test_recall_at_k_returns_zero_when_relevant_chunk_is_missing():
    score = recall_at_k(
        retrieved_chunk_ids=[
            "page-1-chunk-0",
            "page-3-chunk-0",
        ],
        relevant_chunk_ids={"page-2-chunk-0"},
        k=2,
    )

    assert score == 0.0


def test_recall_at_k_handles_multiple_relevant_chunks():
    score = recall_at_k(
        retrieved_chunk_ids=[
            "page-1-chunk-0",
            "page-2-chunk-0",
            "page-3-chunk-0",
            "page-4-chunk-0",
        ],
        relevant_chunk_ids={
            "page-1-chunk-0",
            "page-3-chunk-0",
        },
        k=3,
    )

    assert score == 1.0


def test_reciprocal_rank_returns_one_for_first_result():
    score = reciprocal_rank(
        retrieved_chunk_ids=[
            "page-1-chunk-0",
            "page-2-chunk-0",
        ],
        relevant_chunk_ids={"page-1-chunk-0"},
    )

    assert score == 1.0


def test_reciprocal_rank_returns_half_for_second_result():
    score = reciprocal_rank(
        retrieved_chunk_ids=[
            "page-1-chunk-0",
            "page-2-chunk-0",
            "page-3-chunk-0",
        ],
        relevant_chunk_ids={"page-2-chunk-0"},
    )

    assert score == 0.5


def test_reciprocal_rank_returns_zero_when_not_found():
    score = reciprocal_rank(
        retrieved_chunk_ids=[
            "page-1-chunk-0",
            "page-2-chunk-0",
        ],
        relevant_chunk_ids={"page-3-chunk-0"},
    )

    assert score == 0.0


def test_mean_reciprocal_rank_calculates_average():
    queries = [
        EvaluationQuery(
            query="health insurance",
            relevant_chunk_ids={"page-1-chunk-0"},
        ),
        EvaluationQuery(
            query="vacation policy",
            relevant_chunk_ids={"page-2-chunk-0"},
        ),
    ]

    retrieved_results = [
        [
            "page-1-chunk-0",
            "page-3-chunk-0",
        ],
        [
            "page-3-chunk-0",
            "page-2-chunk-0",
        ],
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
            relevant_chunk_ids={"page-1-chunk-0"},
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
            retrieved_chunk_ids=["page-1-chunk-0"],
            relevant_chunk_ids={"page-1-chunk-0"},
            k=0,
        )

def test_precision_at_k_calculates_correct_score():
    score = precision_at_k(
        retrieved_chunk_ids=[
            "chunk-1",
            "chunk-2",
            "chunk-3",
            "chunk-4",
        ],
        relevant_chunk_ids={"chunk-1", "chunk-3"},
        k=3,
    )

    assert score == pytest.approx(2 / 3)


def test_precision_at_k_returns_one_when_all_results_are_relevant():
    score = precision_at_k(
        retrieved_chunk_ids=["chunk-1", "chunk-2"],
        relevant_chunk_ids={"chunk-1", "chunk-2"},
        k=2,
    )

    assert score == 1.0


def test_precision_at_k_returns_zero_when_no_results_are_relevant():
    score = precision_at_k(
        retrieved_chunk_ids=["chunk-1", "chunk-2"],
        relevant_chunk_ids={"chunk-3"},
        k=2,
    )

    assert score == 0.0


def test_precision_at_k_returns_zero_for_empty_results():
    assert precision_at_k(
        retrieved_chunk_ids=[],
        relevant_chunk_ids={"chunk-1"},
        k=5,
    ) == 0.0


def test_precision_at_k_uses_actual_result_count():
    score = precision_at_k(
        retrieved_chunk_ids=["chunk-1", "chunk-2"],
        relevant_chunk_ids={"chunk-1"},
        k=5,
    )

    assert score == 0.5


def test_precision_at_k_rejects_invalid_k():
    with pytest.raises(ValueError, match="k"):
        precision_at_k(
            retrieved_chunk_ids=["chunk-1"],
            relevant_chunk_ids={"chunk-1"},
            k=0,
        )
