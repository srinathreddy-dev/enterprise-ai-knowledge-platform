from dataclasses import dataclass

from app.ingestion.chunker import TextChunk
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.semantic_search import SemanticSearcher


@dataclass(frozen=True)
class HybridRetrievalResult:
    """A chunk ranked using both lexical and semantic signals."""

    chunk: TextChunk
    score: float
    bm25_score: float
    semantic_score: float


class HybridRetriever:
    """Combine BM25 and semantic retrieval using weighted rank fusion."""

    def __init__(
        self,
        chunks: list[TextChunk],
        bm25_retriever: BM25Retriever | None = None,
        semantic_searcher: SemanticSearcher | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")

        self.chunks = chunks
        self.bm25_retriever = bm25_retriever or BM25Retriever(chunks)
        self.semantic_searcher = semantic_searcher or SemanticSearcher(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
    ) -> list[HybridRetrievalResult]:
        """Return chunks ranked using both retrieval signals."""
        if not query.strip():
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if bm25_weight < 0 or semantic_weight < 0:
            raise ValueError("weights cannot be negative")

        if bm25_weight + semantic_weight == 0:
            raise ValueError("at least one retrieval weight must be greater than zero")

        bm25_results = self.bm25_retriever.search(
            query,
            top_k=len(self.chunks),
        )

        semantic_results = self.semantic_searcher.search(
            query,
            top_k=len(self.chunks),
        )

        bm25_scores = {
            result.chunk.chunk_index: result.score
            for result in bm25_results
        }

        semantic_scores = {
            result.chunk.chunk_index: result.score
            for result in semantic_results
        }

        combined_results: list[HybridRetrievalResult] = []

        for chunk in self.chunks:
            bm25_score = bm25_scores.get(chunk.chunk_index, 0.0)
            semantic_score = semantic_scores.get(chunk.chunk_index, 0.0)

            combined_score = (
                bm25_weight * self._normalize_bm25_score(
                    bm25_score,
                    bm25_results,
                )
                + semantic_weight * self._normalize_semantic_score(
                    semantic_score,
                )
            )

            combined_results.append(
                HybridRetrievalResult(
                    chunk=chunk,
                    score=combined_score,
                    bm25_score=bm25_score,
                    semantic_score=semantic_score,
                )
            )

        combined_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return combined_results[:top_k]

    @staticmethod
    def _normalize_bm25_score(
        score: float,
        results,
    ) -> float:
        """Normalize BM25 scores to a 0-1 range."""
        if not results:
            return 0.0

        maximum = max(result.score for result in results)

        if maximum <= 0:
            return 0.0

        return score / maximum

    @staticmethod
    def _normalize_semantic_score(score: float) -> float:
        """Convert cosine similarity from -1..1 into 0..1."""
        return (score + 1.0) / 2.0
