
from app.evaluation.retrieval_metrics import EvaluationQuery


EVALUATION_QUERIES = [
    EvaluationQuery(
        query="How do employees enroll in health insurance?",
        relevant_chunk_ids={
            "employee_handbook-page-2-chunk-0"
        },
    ),
    EvaluationQuery(
        query="How many paid vacation days do employees receive?",
        relevant_chunk_ids={
            "employee_handbook-page-1-chunk-0"
        },
    ),
    EvaluationQuery(
        query="What retirement savings benefits are available?",
        relevant_chunk_ids={
            "employee_handbook-page-3-chunk-0"
        },
    ),
]

