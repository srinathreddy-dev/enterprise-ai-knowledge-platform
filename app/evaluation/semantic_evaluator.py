
import re
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from sentence_transformers import CrossEncoder


NLIModelLabel = Literal["entailment", "contradiction", "neutral"]

CITATION_PATTERN = re.compile(
    r"\[Document:\s*[^,\]]+,\s*Page:\s*\d+\]",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ClaimEvaluation:
    claim: str
    label: NLIModelLabel
    confidence: float


@dataclass(frozen=True)
class SemanticEvaluationResult:
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    claims: tuple[ClaimEvaluation, ...]


class SemanticFaithfulnessEvaluator:
    """Evaluate answer claims against individual evidence sentences."""

    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"

    LABELS = ("entailment", "contradiction", "neutral")

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.model = CrossEncoder(model_name, device=device)
        self.label_mapping = self._get_label_mapping()

    def _get_label_mapping(self) -> dict[int, NLIModelLabel]:
        mapping: dict[int, NLIModelLabel] = {}

        for index, label in self.model.model.config.id2label.items():
            normalized = str(label).lower()

            if "entail" in normalized:
                mapping[int(index)] = "entailment"
            elif "contrad" in normalized:
                mapping[int(index)] = "contradiction"
            elif "neutral" in normalized:
                mapping[int(index)] = "neutral"

        if len(mapping) != 3:
            raise ValueError(
                "The NLI model must expose entailment, contradiction, "
                "and neutral labels."
            )

        return mapping

    @staticmethod
    def _split_claims(answer: str) -> list[str]:
        """Remove citations and split the answer into individual claims."""
        answer_without_citations = CITATION_PATTERN.sub("", answer).strip()

        sentences = re.split(
            r"(?<=[.!?])\s+",
            answer_without_citations,
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    @staticmethod
    def _prepare_evidence(
        evidence: str | Sequence[str],
    ) -> list[str]:
        """Split supplied evidence chunks into individual sentences."""
        if isinstance(evidence, str):
            evidence_items = [evidence]
        else:
            evidence_items = [
                item for item in evidence
                if isinstance(item, str)
            ]

        sentences = []

        for item in evidence_items:
            item = CITATION_PATTERN.sub("", item).strip()

            parts = re.split(r"(?<=[.!?])\s+", item)

            sentences.extend(
                part.strip()
                for part in parts
                if part.strip()
            )

        return sentences

    def evaluate(
        self,
        answer: str,
        evidence: str | Sequence[str],
    ) -> SemanticEvaluationResult:
        if not answer.strip():
            raise ValueError("Answer cannot be empty")

        evidence_sentences = self._prepare_evidence(evidence)

        if not evidence_sentences:
            raise ValueError("Evidence cannot be empty")

        claims = self._split_claims(answer)

        if not claims:
            raise ValueError("Answer must contain at least one claim")

        pairs = [
            (evidence_sentence, claim)
            for claim in claims
            for evidence_sentence in evidence_sentences
        ]

        probabilities = self.model.predict(
            pairs,
            activation_fn=torch.nn.Softmax(dim=1),
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        probabilities = probabilities.detach().cpu().tolist()

        claim_evaluations = []
        class_probability_totals = {
            label: 0.0 for label in self.LABELS
        }

        for claim_index, claim in enumerate(claims):
            start = claim_index * len(evidence_sentences)
            end = start + len(evidence_sentences)

            claim_scores = probabilities[start:end]

            best_scores = {
                label: max(
                    float(scores[index])
                    for scores in claim_scores
                    for index, mapped_label in self.label_mapping.items()
                    if mapped_label == label
                )
                for label in self.LABELS
            }

            predicted_label = max(
                self.LABELS,
                key=lambda label: best_scores[label],
            )

            confidence = best_scores[predicted_label]

            for label in self.LABELS:
                class_probability_totals[label] += best_scores[label]

            claim_evaluations.append(
                ClaimEvaluation(
                    claim=claim,
                    label=predicted_label,
                    confidence=round(confidence, 4),
                )
            )

        claim_count = len(claim_evaluations)

        average_scores = {
            label: round(total / claim_count, 4)
            for label, total in class_probability_totals.items()
        }

        return SemanticEvaluationResult(
            entailment_score=average_scores["entailment"],
            contradiction_score=average_scores["contradiction"],
            neutral_score=average_scores["neutral"],
            claims=tuple(claim_evaluations),
        )
