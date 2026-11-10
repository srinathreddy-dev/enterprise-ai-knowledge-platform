
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationQuery:
    query: str
    relevant_chunk_ids: set[str]


EVALUATION_QUERIES = [
    EvaluationQuery(
        query="What health insurance benefits are available to employees?",
        relevant_chunk_ids={
            "employee_benefits-page-1-chunk-0",
        },
    ),
    EvaluationQuery(
        query="How many vacation days do employees receive each year?",
        relevant_chunk_ids={
            "leave_attendance_policy-page-1-chunk-0",
        },
    ),
    EvaluationQuery(
        query="What retirement savings benefits does the company offer?",
        relevant_chunk_ids={
            "employee_benefits-page-1-chunk-2",
            "employee_handbook-page-3-chunk-0",
        },
    ),
    EvaluationQuery(
        query="When do employees become eligible for health insurance?",
        relevant_chunk_ids={
            "employee_benefits-page-1-chunk-0",
        },
    ),
    EvaluationQuery(
        query="What dental and vision benefits are available?",
        relevant_chunk_ids={
            "employee_benefits-page-1-chunk-1",
        },
    ),
    EvaluationQuery(
        query="How do employees enroll in the retirement savings plan?",
        relevant_chunk_ids={
            "employee_benefits-page-1-chunk-2",
            "employee_handbook-page-3-chunk-0",
        },
    ),
    EvaluationQuery(
        query="What are the password requirements for company systems?",
        relevant_chunk_ids={
            "it_security_policy-page-1-chunk-0",
        },
    ),
    EvaluationQuery(
        query="Is multi-factor authentication required for company systems?",
        relevant_chunk_ids={
            "it_security_policy-page-1-chunk-1",
        },
    ),
    EvaluationQuery(
        query="How is access to company systems controlled?",
        relevant_chunk_ids={
            "it_security_policy-page-1-chunk-1",
            "it_security_policy-page-1-chunk-2",
        },
    ),
    EvaluationQuery(
        query="What should employees do when they are sick and cannot work?",
        relevant_chunk_ids={
            "leave_attendance_policy-page-1-chunk-1",
        },
    ),
    EvaluationQuery(
        query="How far in advance should employees request vacation?",
        relevant_chunk_ids={
            "leave_attendance_policy-page-1-chunk-0",
            "employee_handbook-page-1-chunk-0",
        },
    ),
    EvaluationQuery(
        query="What are the company's attendance expectations?",
        relevant_chunk_ids={
            "leave_attendance_policy-page-1-chunk-1",
            "leave_attendance_policy-page-1-chunk-2",
        },
    ),
]

