
from app.evaluation.dataset import EVALUATION_QUERIES


def test_evaluation_dataset_contains_twelve_queries():
    assert len(EVALUATION_QUERIES) == 12


def test_evaluation_queries_have_relevant_chunk_ids():
    for query in EVALUATION_QUERIES:
        assert query.query.strip()
        assert query.relevant_chunk_ids
        assert all(
            isinstance(chunk_id, str) and chunk_id.strip()
            for chunk_id in query.relevant_chunk_ids
        )


def test_evaluation_queries_use_expected_chunk_ids():
    chunk_ids = {
        chunk_id
        for query in EVALUATION_QUERIES
        for chunk_id in query.relevant_chunk_ids
    }

    expected_chunk_ids = {
        "employee_handbook-page-1-chunk-0",
        "employee_handbook-page-3-chunk-0",
        "employee_benefits-page-1-chunk-0",
        "employee_benefits-page-1-chunk-1",
        "employee_benefits-page-1-chunk-2",
        "it_security_policy-page-1-chunk-0",
        "it_security_policy-page-1-chunk-1",
        "it_security_policy-page-1-chunk-2",
        "leave_attendance_policy-page-1-chunk-0",
        "leave_attendance_policy-page-1-chunk-1",
        "leave_attendance_policy-page-1-chunk-2",
    }

    assert expected_chunk_ids.issubset(chunk_ids)

