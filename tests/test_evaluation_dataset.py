from app.evaluation.dataset import EVALUATION_QUERIES


def test_evaluation_dataset_contains_expected_queries():
    assert len(EVALUATION_QUERIES) == 3


def test_evaluation_queries_have_relevant_chunks():
    for evaluation_query in EVALUATION_QUERIES:
        assert evaluation_query.query
        assert evaluation_query.relevant_chunk_indexes


def test_evaluation_queries_use_expected_chunk_indexes():
    indexes = [
        next(iter(query.relevant_chunk_indexes))
        for query in EVALUATION_QUERIES
    ]

    assert indexes == [0, 1, 2]
