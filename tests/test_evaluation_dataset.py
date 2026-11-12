
from app.evaluation.dataset import EVALUATION_QUERIES


def test_evaluation_dataset_contains_thirty_queries():
    assert len(EVALUATION_QUERIES) == 30


def test_evaluation_dataset_has_no_duplicate_queries():
    query_texts = [
        query.query
        for query in EVALUATION_QUERIES
    ]

    assert len(query_texts) == len(set(query_texts))


def test_evaluation_dataset_queries_have_relevant_chunks():
    for query in EVALUATION_QUERIES:
        assert query.query.strip()
        assert query.relevant_chunk_ids

