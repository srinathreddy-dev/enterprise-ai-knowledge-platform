
from app.evaluation.dataset import EVALUATION_QUERIES


def test_evaluation_dataset_contains_queries():
    assert len(EVALUATION_QUERIES) == 3


def test_evaluation_queries_have_relevant_chunk_ids():
    for evaluation_query in EVALUATION_QUERIES:
        assert evaluation_query.query
        assert evaluation_query.relevant_chunk_ids


def test_evaluation_queries_use_expected_chunk_ids():
    chunk_ids = [
        next(iter(query.relevant_chunk_ids))
        for query in EVALUATION_QUERIES
    ]

    assert chunk_ids == [
        "employee_handbook-page-2-chunk-0",
        "employee_handbook-page-1-chunk-0",
        "employee_handbook-page-3-chunk-0",
    ]

