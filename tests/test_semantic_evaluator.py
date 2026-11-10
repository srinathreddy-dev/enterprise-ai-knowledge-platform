
import pytest
import torch

from types import SimpleNamespace

from app.evaluation.semantic_evaluator import (
    SemanticFaithfulnessEvaluator,
)


class FakeCrossEncoder:
    """Mock NLI model for isolated unit tests."""

    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

        self.model = SimpleNamespace(
            config=SimpleNamespace(
                id2label={
                    0: "contradiction",
                    1: "entailment",
                    2: "neutral",
                }
            )
        )

    def predict(
        self,
        pairs,
        activation_fn=None,
        convert_to_tensor=False,
        show_progress_bar=False,
    ):
        return torch.tensor(
            [
                [0.05, 0.90, 0.05]
                for _ in pairs
            ]
        )


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.setattr(
        "app.evaluation.semantic_evaluator.CrossEncoder",
        FakeCrossEncoder,
    )

    return SemanticFaithfulnessEvaluator(
        model_name="test-model",
        device="cpu",
    )


def test_evaluator_initializes_with_mock_model(evaluator):
    assert evaluator.model.model_name == "test-model"
    assert evaluator.model.device == "cpu"


def test_evaluator_identifies_entailment(evaluator):
    result = evaluator.evaluate(
        answer="Employees receive fifteen paid vacation days.",
        evidence="Employees receive fifteen paid vacation days each year.",
    )

    assert result.entailment_score == 0.90
    assert result.claims[0].label == "entailment"


def test_evaluator_returns_claim_confidence(evaluator):
    result = evaluator.evaluate(
        answer="Employees receive fifteen paid vacation days.",
        evidence="Employees receive fifteen paid vacation days each year.",
    )

    assert result.claims[0].confidence == 0.90


def test_evaluator_supports_multiple_claims(evaluator):
    result = evaluator.evaluate(
        answer=(
            "Employees receive fifteen paid vacation days. "
            "Vacation requests require manager approval."
        ),
        evidence=(
            "Employees receive fifteen paid vacation days each year. "
            "Vacation requests require manager approval."
        ),
    )

    assert len(result.claims) == 2


def test_evaluator_accepts_multiple_evidence_chunks(evaluator):
    result = evaluator.evaluate(
        answer="Employees receive fifteen paid vacation days.",
        evidence=[
            "Employees receive fifteen paid vacation days.",
            "Vacation requests require manager approval.",
        ],
    )

    assert result.entailment_score == 0.90


def test_evaluator_rejects_empty_answer(evaluator):
    with pytest.raises(ValueError, match="Answer"):
        evaluator.evaluate(
            answer=" ",
            evidence="Some evidence.",
        )


def test_evaluator_rejects_empty_evidence(evaluator):
    with pytest.raises(ValueError, match="Evidence"):
        evaluator.evaluate(
            answer="Some answer.",
            evidence=" ",
        )
