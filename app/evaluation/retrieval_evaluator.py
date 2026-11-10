
from dataclasses import dataclass

from app.evaluation.dataset import EVALUATION_QUERIES
from app.evaluation.retrieval_metrics import (
    EvaluationQuery,
    mean_reciprocal_rank,
    recall_at_k,
)
from app.ingestion.chunker import TextChunk
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.rrf_retriever import RRFHybridRetriever
from app.retrieval.semantic_search import SemanticSearcher


@dataclass(frozen=True)
class RetrievalEvaluation:
    """Evaluation metrics for one retrieval strategy."""

    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mean_reciprocal_rank: float


class RetrievalEvaluator:
    """Evaluate retrieval strategies against a labeled query set."""

    def __init__(
        self,
        chunks: list[TextChunk],
        evaluation_queries: list[EvaluationQuery] | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")

        self.chunks = chunks
        self.evaluation_queries = (
            evaluation_queries
            if evaluation_queries is not None
            else EVALUATION_QUERIES
        )

        self.bm25_retriever = BM25Retriever(chunks)
        self.semantic_searcher = SemanticSearcher(chunks)

        self.hybrid_retriever = HybridRetriever(
            chunks,
            bm25_retriever=self.bm25_retriever,
            semantic_searcher=self.semantic_searcher,
        )

        self.rrf_retriever = RRFHybridRetriever(
            chunks,
            bm25_retriever=self.bm25_retriever,
            semantic_searcher=self.semantic_searcher,
        )

        # Load the cross-encoder only when reranking is requested.
        self.cross_encoder_reranker = None

    def evaluate_bm25(self, k: int = 5) -> RetrievalEvaluation:
        retrieved_results = [
            [
                result.chunk.chunk_id
                for result in self.bm25_retriever.search(query.query, k)
            ]
            for query in self.evaluation_queries
        ]
        return self._calculate_metrics(retrieved_results)

    def evaluate_semantic(self, k: int = 5) -> RetrievalEvaluation:
        retrieved_results = [
            [
                result.chunk.chunk_id
                for result in self.semantic_searcher.search(query.query, k)
            ]
            for query in self.evaluation_queries
        ]
        return self._calculate_metrics(retrieved_results)

    def evaluate_hybrid(self, k: int = 5) -> RetrievalEvaluation:
        retrieved_results = [
            [
                result.chunk.chunk_id
                for result in self.hybrid_retriever.search(query.query, k)
            ]
            for query in self.evaluation_queries
        ]
        return self._calculate_metrics(retrieved_results)

    def evaluate_rrf(self, k: int = 5) -> RetrievalEvaluation:
        return self.evaluate_weighted_rrf(
            k=k,
            bm25_weight=0.70,
            semantic_weight=0.30,
        )

    def evaluate_weighted_rrf(
        self,
        k: int = 5,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> RetrievalEvaluation:
        retriever = RRFHybridRetriever(
            self.chunks,
            bm25_retriever=self.bm25_retriever,
            semantic_searcher=self.semantic_searcher,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
        )

        retrieved_results = [
            [
                result.chunk.chunk_id
                for result in retriever.search(query.query, k)
            ]
            for query in self.evaluation_queries
        ]

        return self._calculate_metrics(retrieved_results)

    def evaluate_rrf_with_reranking(
        self,
        k: int = 5,
        candidate_k: int = 10,
        bm25_weight: float = 0.70,
        semantic_weight: float = 0.30,
        rrf_k: int = 60,
        reranker: CrossEncoderReranker | None = None,
    ) -> RetrievalEvaluation:
        """Evaluate weighted RRF followed by cross-encoder reranking."""

        if k <= 0:
            raise ValueError("k must be greater than zero")

        if candidate_k < k:
            raise ValueError("candidate_k must be at least k")

        if not self.evaluation_queries:
            return self._calculate_metrics([])

        active_reranker = reranker

        if active_reranker is None:
            if self.cross_encoder_reranker is None:
                self.cross_encoder_reranker = CrossEncoderReranker()

            active_reranker = self.cross_encoder_reranker

        retriever = RRFHybridRetriever(
            self.chunks,
            bm25_retriever=self.bm25_retriever,
            semantic_searcher=self.semantic_searcher,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
        )

        retrieved_results = []

        for query in self.evaluation_queries:
            candidates = retriever.search(
                query.query,
                top_k=min(candidate_k, len(self.chunks)),
            )

            reranked_results = active_reranker.rerank(
                query=query.query,
                chunks=[result.chunk for result in candidates],
                top_k=k,
            )

            retrieved_results.append(
                [
                    result.chunk.chunk_id
                    for result in reranked_results
                ]
            )

        return self._calculate_metrics(retrieved_results)

    def _calculate_metrics(
        self,
        retrieved_results: list[list[str]],
    ) -> RetrievalEvaluation:
        if not self.evaluation_queries:
            return RetrievalEvaluation(
                recall_at_1=0.0,
                recall_at_3=0.0,
                recall_at_5=0.0,
                mean_reciprocal_rank=0.0,
            )

        def calculate_average_recall(cutoff: int) -> float:
            scores = [
                recall_at_k(
                    retrieved_chunk_ids,
                    query.relevant_chunk_ids,
                    cutoff,
                )
                for query, retrieved_chunk_ids in zip(
                    self.evaluation_queries,
                    retrieved_results,
                )
            ]
            return sum(scores) / len(scores)

        return RetrievalEvaluation(
            recall_at_1=calculate_average_recall(1),
            recall_at_3=calculate_average_recall(3),
            recall_at_5=calculate_average_recall(5),
            mean_reciprocal_rank=mean_reciprocal_rank(
                self.evaluation_queries,
                retrieved_results,
            ),
        )
