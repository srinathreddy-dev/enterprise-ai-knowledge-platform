
from app.evaluation.answer_quality_evaluator import (
    AnswerQualityEvaluator,
)
from app.ingestion.chunker import TextChunk
from app.reranking.cross_encoder_reranker import RerankedResult
from app.retrieval.bm25_retriever import RetrievalResult


def create_evidence(
    text: str,
    document_id: str,
    page_number: int = 1,
) -> list[RerankedResult]:
    """Create evidence for adversarial evaluation tests."""

    chunk = TextChunk(
        text=text,
        page_number=page_number,
        chunk_index=0,
        document_id=document_id,
    )

    return [
        RerankedResult(
            result=RetrievalResult(
                chunk=chunk,
                score=0.8,
            ),
            score=9.5,
        )
    ]


def test_evaluator_detects_incorrect_vacation_days():
    evaluator = AnswerQualityEvaluator()

    evidence = create_evidence(
        text="Employees receive fifteen paid vacation days each year.",
        document_id="employee_handbook",
    )

    result = evaluator.evaluate(
        question="How many vacation days do employees receive?",
        answer=(
            "Employees receive twenty paid vacation days each year. "
            "[Document: employee_handbook, Page: 1]"
        ),
        evidence=evidence,
    )

    assert result.faithfulness_score < 1.0


def test_evaluator_detects_fabricated_password_requirement():
    evaluator = AnswerQualityEvaluator()

    evidence = create_evidence(
        text=(
            "Passwords must contain at least 12 characters "
            "and include letters, numbers, and symbols."
        ),
        document_id="it_security_policy",
    )

    result = evaluator.evaluate(
        question="What are the password requirements?",
        answer=(
            "Passwords must contain at least 12 characters "
            "and include letters, numbers, symbols, and mandatory "
            "30-day rotation."
            "[Document: it_security_policy, Page: 1]"
        ),
        evidence=evidence,
    )

    assert result.faithfulness_score < 1.0


def test_evaluator_detects_irrelevant_answer():
    evaluator = AnswerQualityEvaluator()

    evidence = create_evidence(
        text="Employees receive fifteen paid vacation days each year.",
        document_id="employee_handbook",
    )

    result = evaluator.evaluate(
        question="How many vacation days do employees receive?",
        answer="The company uses cloud applications for business operations.",
        evidence=evidence,
    )

    assert result.answer_relevance_score < 1.0


def test_evaluator_rejects_citation_to_unretrieved_document():
    evaluator = AnswerQualityEvaluator()

    evidence = create_evidence(
        text="Employees receive fifteen paid vacation days each year.",
        document_id="employee_handbook",
    )

    result = evaluator.evaluate(
        question="How many vacation days do employees receive?",
        answer=(
            "Employees receive fifteen paid vacation days. "
            "[Document: finance_policy, Page: 1]"
        ),
        evidence=evidence,
    )

    assert result.citation_accuracy_score == 0.0
