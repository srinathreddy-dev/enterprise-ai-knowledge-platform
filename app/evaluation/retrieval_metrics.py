from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationQuery:
    """A retrieval query and the chunks considered relevant."""

    query: str
    relevant_chunk_ids: set[str]


def recall_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    """Calculate recall among the top-k retrieved chunks."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_chunk_ids:
        return 0.0

    retrieved = set(retrieved_chunk_ids[:k])
    relevant_retrieved = retrieved.intersection(relevant_chunk_ids)

    return len(relevant_retrieved) / len(relevant_chunk_ids)


def reciprocal_rank(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
) -> float:
    """Calculate reciprocal rank of the first relevant result."""
    if not relevant_chunk_ids:
        return 0.0

    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    queries: list[EvaluationQuery],
    retrieved_results: list[list[str]],
) -> float:
    """Calculate mean reciprocal rank across evaluation queries."""
    if not queries:
        return 0.0

    if len(queries) != len(retrieved_results):
        raise ValueError(
            "queries and retrieved_results must have the same length"
        )

    reciprocal_ranks = [
        reciprocal_rank(
            retrieved_chunk_ids,
            evaluation_query.relevant_chunk_ids,
        )
        for evaluation_query, retrieved_chunk_ids in zip(
            queries,
            retrieved_results,
        )
    ]

    return sum(reciprocal_ranks) / len(reciprocal_ranks)

def precision_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    """Calculate precision among the top-k retrieved chunks."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    retrieved = retrieved_chunk_ids[:k]

    if not retrieved:
        return 0.0

    relevant_retrieved = sum(
        chunk_id in relevant_chunk_ids
        for chunk_id in retrieved
    )

    return relevant_retrieved / len(retrieved)
