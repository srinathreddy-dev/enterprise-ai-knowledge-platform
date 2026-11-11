
from dataclasses import dataclass

from app.evaluation.dataset import EVALUATION_QUERIES
from app.evaluation.retrieval_metrics import (
    EvaluationQuery,
    mean_reciprocal_rank,
    recall_at_k,
)
from app.ingestion.chunker import TextChunk
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.bm25_retriever import (
    BM25Retriever,
    RetrievalResult,
)
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


@dataclass(frozen=True)
class QueryDiagnostic:
    """Per-query retrieval diagnostics."""

    query: str
    relevant_chunk_ids: set[str]
    retrieved_chunk_ids: list[str]
    first_result_relevant: bool
    missed_relevant_chunk_ids: set[str]
    first_relevant_rank: int | None


class RetrievalEvaluator:
    """Evaluate retrieval strategies against a labeled query set."""

    def __init__(
        self,
        chunks: list[TextChunk],
        evaluation_queries: list[EvaluationQuery] | None = None,
        embedder=None,
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

        self.semantic_searcher = SemanticSearcher(
            chunks,
            embedder=embedder,
        )

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

        self.cross_encoder_reranker = CrossEncoderReranker()

    def evaluate_bm25(
        self,
        k: int = 5,
    ) -> RetrievalEvaluation:
        retrieved_results = [
            [
                result.chunk.chunk_id
                for result in self.bm25_retriever.search(
                    query.query,
                    k,
                )
            ]
            for query in self.evaluation_queries
        ]

        return self._calculate_metrics(retrieved_results)

    def evaluate_semantic(
        self,
        k: int = 5,
    ) -> RetrievalEvaluation:
        retrieved_results = [
            [
                result.chunk.chunk_id
                for result in self.semantic_searcher.search(
                    query.query,
                    k,
                )
            ]
            for query in self.evaluation_queries
        ]

        return self._calculate_metrics(retrieved_results)

    def evaluate_hybrid(
        self,
        k: int = 5,
    ) -> RetrievalEvaluation:
        retrieved_results = [
            [
                result.chunk.chunk_id
                for result in self.hybrid_retriever.search(
                    query.query,
                    k,
                )
            ]
            for query in self.evaluation_queries
        ]

        return self._calculate_metrics(retrieved_results)

    def evaluate_hybrid_with_reranking(
        self,
        k: int = 5,
        candidate_k: int = 10,
        reranker: CrossEncoderReranker | None = None,
    ) -> RetrievalEvaluation:
        """Evaluate hybrid retrieval followed by production reranking."""

        self._validate_cutoffs(k, candidate_k)

        if not self.evaluation_queries:
            return self._calculate_metrics([])

        active_reranker = reranker or self.cross_encoder_reranker

        retrieved_results = []

        for query in self.evaluation_queries:
            candidates = self.hybrid_retriever.search(
                query=query.query,
                top_k=min(candidate_k, len(self.chunks)),
            )

            candidate_results = [
                RetrievalResult(
                    chunk=result.chunk,
                    score=result.score,
                )
                for result in candidates
            ]

            reranked_results = active_reranker.rerank(
                query=query.query,
                results=candidate_results,
                top_k=k,
            )

            retrieved_results.append([
                result.result.chunk.chunk_id
                for result in reranked_results
            ])

        return self._calculate_metrics(retrieved_results)

    def evaluate_rrf(
        self,
        k: int = 5,
    ) -> RetrievalEvaluation:
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
                for result in retriever.search(
                    query.query,
                    k,
                )
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
        """Evaluate weighted RRF followed by production reranking."""

        self._validate_cutoffs(k, candidate_k)

        if not self.evaluation_queries:
            return self._calculate_metrics([])

        active_reranker = reranker or self.cross_encoder_reranker

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

            candidate_results = [
                RetrievalResult(
                    chunk=result.chunk,
                    score=result.score,
                )
                for result in candidates
            ]

            reranked_results = active_reranker.rerank(
                query=query.query,
                results=candidate_results,
                top_k=k,
            )

            retrieved_results.append([
                result.result.chunk.chunk_id
                for result in reranked_results
            ])

        return self._calculate_metrics(retrieved_results)

    def diagnose(
        self,
        strategy: str = "hybrid_reranked",
        k: int = 5,
        candidate_k: int = 10,
    ) -> list[QueryDiagnostic]:
        """
        Return per-question diagnostics for a retrieval strategy.

        Supported strategies:
        bm25, semantic, hybrid, hybrid_reranked,
        rrf, rrf_reranked.
        """

        if k <= 0:
            raise ValueError("k must be greater than zero")

        if candidate_k < k:
            raise ValueError("candidate_k must be at least k")

        retrieved_results = self._retrieve_for_strategy(
            strategy=strategy,
            k=k,
            candidate_k=candidate_k,
        )

        diagnostics = []

        for query, retrieved_ids in zip(
            self.evaluation_queries,
            retrieved_results,
        ):
            relevant_ids = query.relevant_chunk_ids

            matched_ids = set(retrieved_ids) & relevant_ids

            first_relevant_rank = next(
                (
                    rank
                    for rank, chunk_id in enumerate(
                        retrieved_ids,
                        start=1,
                    )
                    if chunk_id in relevant_ids
                ),
                None,
            )

            diagnostics.append(
                QueryDiagnostic(
                    query=query.query,
                    relevant_chunk_ids=set(relevant_ids),
                    retrieved_chunk_ids=list(retrieved_ids),
                    first_result_relevant=(
                        bool(retrieved_ids)
                        and retrieved_ids[0] in relevant_ids
                    ),
                    missed_relevant_chunk_ids=(
                        relevant_ids - matched_ids
                    ),
                    first_relevant_rank=first_relevant_rank,
                )
            )

        return diagnostics

    def _retrieve_for_strategy(
        self,
        strategy: str,
        k: int,
        candidate_k: int,
    ) -> list[list[str]]:
        """Retrieve ranked chunk IDs for a selected strategy."""

        results = []

        for query in self.evaluation_queries:
            if strategy == "bm25":
                retrieved = self.bm25_retriever.search(
                    query.query,
                    k,
                )

                chunk_ids = [
                    result.chunk.chunk_id
                    for result in retrieved
                ]

            elif strategy == "semantic":
                retrieved = self.semantic_searcher.search(
                    query.query,
                    k,
                )

                chunk_ids = [
                    result.chunk.chunk_id
                    for result in retrieved
                ]

            elif strategy == "hybrid":
                retrieved = self.hybrid_retriever.search(
                    query.query,
                    k,
                )

                chunk_ids = [
                    result.chunk.chunk_id
                    for result in retrieved
                ]

            elif strategy == "hybrid_reranked":
                candidates = self.hybrid_retriever.search(
                    query.query,
                    min(candidate_k, len(self.chunks)),
                )

                candidate_results = [
                    RetrievalResult(
                        chunk=result.chunk,
                        score=result.score,
                    )
                    for result in candidates
                ]

                reranked = self.cross_encoder_reranker.rerank(
                    query=query.query,
                    results=candidate_results,
                    top_k=k,
                )

                chunk_ids = [
                    result.result.chunk.chunk_id
                    for result in reranked
                ]

            elif strategy in {"rrf", "rrf_reranked"}:
                retriever = self.rrf_retriever

                candidates = retriever.search(
                    query.query,
                    (
                        min(candidate_k, len(self.chunks))
                        if strategy == "rrf_reranked"
                        else k
                    ),
                )

                if strategy == "rrf_reranked":
                    candidate_results = [
                        RetrievalResult(
                            chunk=result.chunk,
                            score=result.score,
                        )
                        for result in candidates
                    ]

                    reranked = self.cross_encoder_reranker.rerank(
                        query=query.query,
                        results=candidate_results,
                        top_k=k,
                    )

                    chunk_ids = [
                        result.result.chunk.chunk_id
                        for result in reranked
                    ]
                else:
                    chunk_ids = [
                        result.chunk.chunk_id
                        for result in candidates
                    ]

            else:
                raise ValueError(
                    f"Unsupported retrieval strategy: {strategy}"
                )

            results.append(chunk_ids)

        return results

    @staticmethod
    def _validate_cutoffs(
        k: int,
        candidate_k: int,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be greater than zero")

        if candidate_k < k:
            raise ValueError("candidate_k must be at least k")

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

        if len(retrieved_results) != len(self.evaluation_queries):
            raise ValueError(
                "Retrieved results must match the number of evaluation queries"
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
