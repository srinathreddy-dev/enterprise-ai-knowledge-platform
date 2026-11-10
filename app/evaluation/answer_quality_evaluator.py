
import re
from dataclasses import dataclass

import numpy as np

from app.evaluation.semantic_evaluator import (
    SemanticFaithfulnessEvaluator,
)
from app.reranking.cross_encoder_reranker import RerankedResult
from app.retrieval.semantic_retriever import SemanticEmbedder


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "for", "from", "has", "have", "in", "is", "it",
    "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "will", "with",
}

CITATION_PATTERN = re.compile(
    r"\[Document:\s*([^,\]]+),\s*Page:\s*(\d+)\]",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class AnswerQualityResult:
    faithfulness_score: float
    answer_relevance_score: float
    citation_accuracy_score: float
    semantic_faithfulness_score: float | None = None
    semantic_relevance_score: float | None = None


class AnswerQualityEvaluator:
    """Evaluate answer quality using lexical and semantic metrics."""

    def __init__(
        self,
        semantic_evaluator: SemanticFaithfulnessEvaluator | None = None,
        semantic_embedder: SemanticEmbedder | None = None,
    ):
        self.semantic_evaluator = semantic_evaluator
        self.semantic_embedder = semantic_embedder

    @staticmethod
    def _remove_citations(text: str) -> str:
        return CITATION_PATTERN.sub("", text)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

        return {
            token
            for token in tokens
            if token not in STOP_WORDS
        }

    def evaluate(
        self,
        question: str,
        answer: str,
        evidence: list[RerankedResult],
    ) -> AnswerQualityResult:

        if not question.strip():
            raise ValueError("Question cannot be empty")

        if not answer.strip():
            raise ValueError("Answer cannot be empty")

        if not evidence:
            raise ValueError("Evidence cannot be empty")

        semantic_faithfulness = None
        semantic_relevance = None

        # Semantic faithfulness: Does evidence support the answer?
        if self.semantic_evaluator is not None:
            evidence_texts = [
                item.result.chunk.text
                for item in evidence
            ]

            faithfulness_result = self.semantic_evaluator.evaluate(
                answer=answer,
                evidence=evidence_texts,
            )

            semantic_faithfulness = (
                faithfulness_result.entailment_score
            )

        # Semantic relevance: Does the answer address the question?
        if self.semantic_embedder is not None:
            clean_answer = self._remove_citations(answer)

            question_embedding = (
                self.semantic_embedder.embed_query(question)
            )

            answer_embedding = (
                self.semantic_embedder.embed_query(clean_answer)
            )

            similarity = float(
                np.dot(question_embedding, answer_embedding)
            )

            # Keep the metric within the range [0, 1].
            semantic_relevance = round(
                float(np.clip(similarity, 0.0, 1.0)),
                4,
            )

        return AnswerQualityResult(
            faithfulness_score=self._faithfulness_score(
                answer,
                evidence,
            ),
            answer_relevance_score=self._answer_relevance_score(
                question,
                answer,
            ),
            citation_accuracy_score=self._citation_accuracy_score(
                answer,
                evidence,
            ),
            semantic_faithfulness_score=semantic_faithfulness,
            semantic_relevance_score=semantic_relevance,
        )

    def _faithfulness_score(
        self,
        answer: str,
        evidence: list[RerankedResult],
    ) -> float:

        answer_tokens = self._tokenize(
            self._remove_citations(answer)
        )

        evidence_text = " ".join(
            item.result.chunk.text
            for item in evidence
        )

        evidence_tokens = self._tokenize(evidence_text)

        if not answer_tokens:
            return 0.0

        supported_tokens = answer_tokens.intersection(
            evidence_tokens
        )

        return round(
            len(supported_tokens) / len(answer_tokens),
            4,
        )

    def _answer_relevance_score(
        self,
        question: str,
        answer: str,
    ) -> float:

        question_tokens = self._tokenize(question)
        answer_tokens = self._tokenize(
            self._remove_citations(answer)
        )

        if not question_tokens:
            return 0.0

        if "many" in question_tokens:
            quantity_words = {
                "zero", "one", "two", "three", "four",
                "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve", "thirteen",
                "fourteen", "fifteen", "sixteen", "seventeen",
                "eighteen", "nineteen", "twenty",
            }

            if (
                answer_tokens.intersection(quantity_words)
                or any(token.isdigit() for token in answer_tokens)
            ):
                question_tokens.discard("many")
                question_tokens.discard("how")
                question_tokens.discard("do")

        relevant_tokens = question_tokens.intersection(
            answer_tokens
        )

        return round(
            len(relevant_tokens) / len(question_tokens),
            4,
        )

    @staticmethod
    def _citation_accuracy_score(
        answer: str,
        evidence: list[RerankedResult],
    ) -> float:

        cited_documents_and_pages = {
            (document_id.strip(), int(page))
            for document_id, page in CITATION_PATTERN.findall(answer)
        }

        if not cited_documents_and_pages:
            return 0.0

        evidence_documents_and_pages = {
            (
                item.result.chunk.document_id,
                item.result.chunk.page_number,
            )
            for item in evidence
        }

        valid_citations = cited_documents_and_pages.intersection(
            evidence_documents_and_pages
        )

        return round(
            len(valid_citations) / len(cited_documents_and_pages),
            4,
        )
