
import pytest

from app.evaluation.answer_quality_evaluator import (
    AnswerQualityEvaluator,
)
from app.ingestion.chunker import TextChunk
from app.reranking.cross_encoder_reranker import RerankedResult
from app.retrieval.bm25_retriever import RetrievalResult


@pytest.fixture
def evaluator():
    return AnswerQualityEvaluator()


@pytest.fixture
def evidence():
    chunk = TextChunk(
        text=(
            "Employees receive fifteen paid vacation days "
            "each year. Vacation requests require manager approval."
        ),
        page_number=2,
        chunk_index=0,
        document_id="employee_handbook",
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


def test_faithfulness_is_one_for_supported_answer(
    evaluator,
    evidence,
):
    answer = (
        "Employees receive fifteen paid vacation days each year. "
        "[Document: employee_handbook, Page: 2]"
    )

    result = evaluator.evaluate(
        question="How many vacation days do employees receive?",
        answer=answer,
        evidence=evidence,
    )

    assert result.faithfulness_score == 1.0


def test_faithfulness_detects_unsupported_terms(
    evaluator,
    evidence,
):
    answer = (
        "Employees receive fifteen paid vacation days "
        "and unlimited bonuses each year."
    )

    result = evaluator.evaluate(
        question="What benefits do employees receive?",
        answer=answer,
        evidence=evidence,
    )

    assert result.faithfulness_score < 1.0


def test_answer_relevance_is_one_for_relevant_answer(
    evaluator,
    evidence,
):
    result = evaluator.evaluate(
        question="How many vacation days do employees receive?",
        answer="Employees receive fifteen vacation days.",
        evidence=evidence,
    )

    assert result.answer_relevance_score == 1.0


def test_answer_relevance_detects_irrelevant_answer(
    evaluator,
    evidence,
):
    result = evaluator.evaluate(
        question="How many vacation days do employees receive?",
        answer="The company provides manager approval.",
        evidence=evidence,
    )

    assert result.answer_relevance_score < 1.0


def test_citation_accuracy_is_one_for_valid_document_and_page(
    evaluator,
    evidence,
):
    result = evaluator.evaluate(
        question="How many vacation days do employees receive?",
        answer=(
            "Employees receive fifteen vacation days. "
            "[Document: employee_handbook, Page: 2]"
        ),
        evidence=evidence,
    )

    assert result.citation_accuracy_score == 1.0


def test_citation_accuracy_is_zero_for_wrong_document(
    evaluator,
    evidence,
):
    result = evaluator.evaluate(
        question="How many vacation days do employees receive?",
        answer=(
            "Employees receive fifteen vacation days. "
            "[Document: it_security_policy, Page: 2]"
        ),
        evidence=evidence,
    )

    assert result.citation_accuracy_score == 0.0


def test_citation_accuracy_is_zero_for_wrong_page(
    evaluator,
    evidence,
):
    result = evaluator.evaluate(
        question="How many vacation days do employees receive?",
        answer=(
            "Employees receive fifteen vacation days. "
            "[Document: employee_handbook, Page: 5]"
        ),
        evidence=evidence,
    )

    assert result.citation_accuracy_score == 0.0


def test_citation_accuracy_is_zero_for_old_page_only_format(
    evaluator,
    evidence,
):
    result = evaluator.evaluate(
        question="How many vacation days do employees receive?",
        answer="Employees receive fifteen vacation days. [Page 2]",
        evidence=evidence,
    )

    assert result.citation_accuracy_score == 0.0


def test_citation_accuracy_is_zero_without_citation(
    evaluator,
    evidence,
):
    result = evaluator.evaluate(
        question="How many vacation days do employees receive?",
        answer="Employees receive fifteen vacation days.",
        evidence=evidence,
    )

    assert result.citation_accuracy_score == 0.0


@pytest.mark.parametrize(
    "question, answer, evidence_value, error",
    [
        ("", "Some answer", True, "Question"),
        ("Some question", "", True, "Answer"),
        ("Some question", "Some answer", False, "Evidence"),
    ],
)
def test_evaluator_rejects_invalid_inputs(
    evaluator,
    evidence,
    question,
    answer,
    evidence_value,
    error,
):
    supplied_evidence = evidence if evidence_value else []

    with pytest.raises(ValueError, match=error):
        evaluator.evaluate(
            question=question,
            answer=answer,
            evidence=supplied_evidence,
        )


from types import SimpleNamespace


class FakeSemanticEvaluator:
    def __init__(self):
        self.received_answer = None
        self.received_evidence = None

    def evaluate(self, answer, evidence):
        self.received_answer = answer
        self.received_evidence = evidence

        return SimpleNamespace(
            entailment_score=0.85,
            contradiction_score=0.10,
            neutral_score=0.05,
        )


def test_semantic_faithfulness_is_none_when_not_configured(
    evaluator,
    evidence,
):
    result = evaluator.evaluate(
        question="How many vacation days do employees receive?",
        answer="Employees receive fifteen vacation days.",
        evidence=evidence,
    )

    assert result.semantic_faithfulness_score is None


def test_semantic_evaluator_is_integrated(evidence):
    fake_semantic_evaluator = FakeSemanticEvaluator()

    evaluator = AnswerQualityEvaluator(
        semantic_evaluator=fake_semantic_evaluator
    )

    answer = "Employees receive fifteen vacation days."

    result = evaluator.evaluate(
        question="How many vacation days do employees receive?",
        answer=answer,
        evidence=evidence,
    )

    assert result.semantic_faithfulness_score == 0.85
    assert fake_semantic_evaluator.received_answer == answer
    assert len(fake_semantic_evaluator.received_evidence) == len(evidence)


def test_semantic_evaluator_receives_evidence_text(evidence):
    fake_semantic_evaluator = FakeSemanticEvaluator()

    evaluator = AnswerQualityEvaluator(
        semantic_evaluator=fake_semantic_evaluator
    )

    evaluator.evaluate(
        question="How many vacation days do employees receive?",
        answer="Employees receive fifteen vacation days.",
        evidence=evidence,
    )

    expected_evidence = [
        item.result.chunk.text
        for item in evidence
    ]

    assert fake_semantic_evaluator.received_evidence == expected_evidence
