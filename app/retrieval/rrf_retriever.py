from dataclasses import dataclass

from app.ingestion.chunker import TextChunk
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.semantic_search import SemanticSearcher


@dataclass(frozen=True)
class RRFResult:
    """A retrieved chunk ranked using Reciprocal Rank Fusion."""

    chunk: TextChunk
    score: float
    bm25_rank: int | None
    semantic_rank: int | None


class RRFHybridRetriever:
    """Combine BM25 and semantic rankings using weighted RRF."""

    def __init__(
        self,
        chunks: list[TextChunk],
        bm25_retriever: BM25Retriever | None = None,
        semantic_searcher: SemanticSearcher | None = None,
        rrf_k: int = 60,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")

        if bm25_weight < 0 or semantic_weight < 0:
            raise ValueError("Weights cannot be negative")

        if bm25_weight + semantic_weight == 0:
            raise ValueError("At least one weight must be greater than zero")

        self.chunks = chunks
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

        self.bm25_retriever = (
            bm25_retriever or BM25Retriever(chunks)
        )
        self.semantic_searcher = (
            semantic_searcher or SemanticSearcher(chunks)
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RRFResult]:
        """Return chunks ranked using weighted Reciprocal Rank Fusion."""

        if not query.strip():
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        bm25_results = self.bm25_retriever.search(
            query,
            top_k=len(self.chunks),
        )

        semantic_results = self.semantic_searcher.search(
            query,
            top_k=len(self.chunks),
        )

        bm25_ranks = {
            result.chunk.chunk_id: rank
            for rank, result in enumerate(bm25_results, start=1)
        }

        semantic_ranks = {
            result.chunk.chunk_id: rank
            for rank, result in enumerate(semantic_results, start=1)
        }

        fused_results = []

        for chunk in self.chunks:
            bm25_rank = bm25_ranks.get(chunk.chunk_id)
            semantic_rank = semantic_ranks.get(chunk.chunk_id)

            score = 0.0

            if bm25_rank is not None:
                score += (
                    self.bm25_weight
                    / (self.rrf_k + bm25_rank)
                )

            if semantic_rank is not None:
                score += (
                    self.semantic_weight
                    / (self.rrf_k + semantic_rank)
                )

            fused_results.append(
                RRFResult(
                    chunk=chunk,
                    score=score,
                    bm25_rank=bm25_rank,
                    semantic_rank=semantic_rank,
                )
            )

        fused_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return fused_results[:top_k]
