
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
            "employee_handbook-page-2-chunk-0",
        },
    ),
    EvaluationQuery(
        query="How many vacation days do employees receive each year?",
        relevant_chunk_ids={
            "leave_attendance_policy-page-1-chunk-0",
            "employee_handbook-page-1-chunk-0",
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
            "employee_handbook-page-2-chunk-0",
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

    EvaluationQuery(
        query="When must new employees complete benefits enrollment?",
        relevant_chunk_ids={
            "employee_benefits-page-2-chunk-0",
        },
    ),
    EvaluationQuery(
        query="When can employees change their benefit elections?",
        relevant_chunk_ids={
            "employee_benefits-page-2-chunk-0",
            "employee_handbook-page-2-chunk-0",
        },
    ),
    EvaluationQuery(
        query="Can employees add dependents to their health insurance?",
        relevant_chunk_ids={
            "employee_benefits-page-1-chunk-1",
        },
    ),
    EvaluationQuery(
        query="Who pays the monthly medical insurance premium?",
        relevant_chunk_ids={
            "employee_benefits-page-1-chunk-0",
            "employee_benefits-page-1-chunk-1",
        },
    ),
    EvaluationQuery(
        query="Do unused vacation days carry over to the next year?",
        relevant_chunk_ids={
            "employee_handbook-page-1-chunk-0",
        },
    ),
    EvaluationQuery(
        query="Who approves paid time off requests?",
        relevant_chunk_ids={
            "leave_attendance_policy-page-1-chunk-0",
            "employee_handbook-page-1-chunk-0",
        },
    ),
    EvaluationQuery(
        query="After how many days can employees join the retirement plan?",
        relevant_chunk_ids={
            "employee_handbook-page-3-chunk-0",
        },
    ),
    EvaluationQuery(
        query="How do employees select their retirement contribution percentage?",
        relevant_chunk_ids={
            "employee_benefits-page-1-chunk-2",
            "employee_handbook-page-3-chunk-0",
        },
    ),
    EvaluationQuery(
        query="What is the minimum password length?",
        relevant_chunk_ids={
            "it_security_policy-page-1-chunk-0",
        },
    ),
    EvaluationQuery(
        query="What should employees do if their password is compromised?",
        relevant_chunk_ids={
            "it_security_policy-page-1-chunk-0",
        },
    ),
    EvaluationQuery(
        query="Which systems require multi-factor authentication?",
        relevant_chunk_ids={
            "it_security_policy-page-1-chunk-1",
        },
    ),
    EvaluationQuery(
        query="Who is responsible for requesting system access for new employees?",
        relevant_chunk_ids={
            "it_security_policy-page-1-chunk-1",
            "it_security_policy-page-1-chunk-2",
        },
    ),
    EvaluationQuery(
        query="How should employees report suspected security incidents?",
        relevant_chunk_ids={
            "it_security_policy-page-2-chunk-0",
        },
    ),
    EvaluationQuery(
        query="How should employees protect company devices and confidential data?",
        relevant_chunk_ids={
            "it_security_policy-page-2-chunk-0",
            "it_security_policy-page-2-chunk-1",
        },
    ),
    EvaluationQuery(
        query="Who should employees contact for an extended leave of absence?",
        relevant_chunk_ids={
            "leave_attendance_policy-page-1-chunk-2",
        },
    ),
    EvaluationQuery(
        query="What should employees do if they expect to arrive late?",
        relevant_chunk_ids={
            "leave_attendance_policy-page-1-chunk-1",
            "leave_attendance_policy-page-1-chunk-2",
        },
    ),
    EvaluationQuery(
        query="Where can employees find the company's holiday schedule?",
        relevant_chunk_ids={
            "leave_attendance_policy-page-2-chunk-0",
        },
    ),
    EvaluationQuery(
        query="Can employees take sick leave when they are unable to work because of illness?",
        relevant_chunk_ids={
            "leave_attendance_policy-page-1-chunk-0",
            "leave_attendance_policy-page-1-chunk-1",
        },
    ),

]

