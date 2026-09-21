from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from app.retrieval.bm25_retriever import RetrievalResult


@dataclass(frozen=True)
class RerankedResult:
    """A retrieved chunk with its cross-encoder relevance score."""

    result: RetrievalResult
    score: float


class CrossEncoderReranker:
    """Rerank retrieved chunks using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        """Rerank retrieval results for a query."""
        if not query.strip():
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not results:
            return []

        pairs = [
            (query, result.chunk.text)
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked_results = [
            RerankedResult(
                result=result,
                score=float(score),
            )
            for result, score in zip(results, scores)
        ]

        reranked_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return reranked_results[:top_k]
