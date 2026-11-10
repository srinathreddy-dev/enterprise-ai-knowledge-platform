
import re
from dataclasses import dataclass

from app.retrieval.bm25_retriever import RetrievalResult


@dataclass(frozen=True)
class RerankedResult:
    """A retrieved chunk with its relevance score."""

    result: RetrievalResult
    score: float


class CrossEncoderReranker:
    """Rerank retrieved chunks using lightweight lexical relevance."""

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        """Rerank results without loading a machine-learning model."""
        if not query.strip():
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not results:
            return []

        query_terms = set(re.findall(r"\b\w+\b", query.lower()))

        reranked_results = []

        for result in results:
            document_terms = set(
                re.findall(r"\b\w+\b", result.chunk.text.lower())
            )

            overlap = len(query_terms & document_terms)
            lexical_score = overlap / max(len(query_terms), 1)

            combined_score = (
                0.5 * lexical_score
                + 0.5 * result.score
            )

            reranked_results.append(
                RerankedResult(
                    result=result,
                    score=float(combined_score),
                )
            )

        reranked_results.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return reranked_results[:top_k]
