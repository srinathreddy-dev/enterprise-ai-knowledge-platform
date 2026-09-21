from dataclasses import dataclass

from app.evaluation.dataset import EVALUATION_QUERIES
from app.evaluation.retrieval_metrics import (
    EvaluationQuery,
    mean_reciprocal_rank,
    recall_at_k,
)
from app.ingestion.chunker import TextChunk
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.semantic_search import SemanticSearcher


@dataclass(frozen=True)
class RetrievalEvaluation:
    """Evaluation metrics for one retrieval strategy."""

    recall_at_k: float
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

    def evaluate_bm25(self, k: int = 5) -> RetrievalEvaluation:
        """Evaluate BM25 retrieval."""
        retrieved_results = [
            [
                result.chunk.chunk_index
                for result in self.bm25_retriever.search(
                    evaluation_query.query,
                    top_k=k,
                )
            ]
            for evaluation_query in self.evaluation_queries
        ]

        return self._calculate_metrics(
            retrieved_results,
            k,
        )

    def evaluate_semantic(self, k: int = 5) -> RetrievalEvaluation:
        """Evaluate semantic retrieval."""
        retrieved_results = [
            [
                result.chunk.chunk_index
                for result in self.semantic_searcher.search(
                    evaluation_query.query,
                    top_k=k,
                )
            ]
            for evaluation_query in self.evaluation_queries
        ]

        return self._calculate_metrics(
            retrieved_results,
            k,
        )

    def evaluate_hybrid(self, k: int = 5) -> RetrievalEvaluation:
        """Evaluate hybrid retrieval."""
        retrieved_results = [
            [
                result.chunk.chunk_index
                for result in self.hybrid_retriever.search(
                    evaluation_query.query,
                    top_k=k,
                )
            ]
            for evaluation_query in self.evaluation_queries
        ]

        return self._calculate_metrics(
            retrieved_results,
            k,
        )

    def _calculate_metrics(
        self,
        retrieved_results: list[list[int]],
        k: int,
    ) -> RetrievalEvaluation:
        recall_scores = [
            recall_at_k(
                retrieved_chunk_indexes,
                evaluation_query.relevant_chunk_indexes,
                k,
            )
            for evaluation_query, retrieved_chunk_indexes in zip(
                self.evaluation_queries,
                retrieved_results,
            )
        ]

        mean_recall = sum(recall_scores) / len(recall_scores)

        mrr = mean_reciprocal_rank(
            self.evaluation_queries,
            retrieved_results,
        )

        return RetrievalEvaluation(
            recall_at_k=mean_recall,
            mean_reciprocal_rank=mrr,
        )
