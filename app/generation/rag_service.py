
from dataclasses import dataclass

from app.generation.answer_service import AnswerGenerationService
from app.ingestion.chunker import TextChunk
from app.reranking.cross_encoder_reranker import (
    CrossEncoderReranker,
    RerankedResult,
)
from app.retrieval.bm25_retriever import RetrievalResult
from app.retrieval.hybrid_retriever import HybridRetriever


@dataclass(frozen=True)
class RAGResponse:
    """The answer and the evidence used to generate it."""

    question: str
    answer: str
    evidence: list[RerankedResult]


class RAGService:
    """Coordinate hybrid retrieval, reranking, and answer generation."""

    def __init__(
        self,
        chunks: list[TextChunk],
        hybrid_retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        answer_service: AnswerGenerationService | None = None,
        retrieval_top_k: int = 10,
        evidence_top_k: int = 5,
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")
        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be greater than zero")
        if evidence_top_k <= 0:
            raise ValueError("evidence_top_k must be greater than zero")

        self.chunks = chunks
        self.hybrid_retriever = hybrid_retriever or HybridRetriever(chunks)
        self.reranker = reranker or CrossEncoderReranker()
        self.answer_service = answer_service or AnswerGenerationService()
        self.retrieval_top_k = retrieval_top_k
        self.evidence_top_k = evidence_top_k

    def ask(self, question: str) -> RAGResponse:
        """Retrieve evidence, rerank it, and generate a grounded answer."""
        if not question.strip():
            raise ValueError("Question cannot be empty")

        hybrid_results = self.hybrid_retriever.search(
            query=question,
            top_k=self.retrieval_top_k,
        )

        retrieval_results = [
            RetrievalResult(
                chunk=result.chunk,
                score=result.score,
            )
            for result in hybrid_results
        ]

        if not retrieval_results:
            raise ValueError("No relevant evidence was retrieved")

        reranked_evidence = self.reranker.rerank(
            query=question,
            results=retrieval_results,
            top_k=self.evidence_top_k,
        )

        if not reranked_evidence:
            raise ValueError("No evidence remained after reranking")

        answer = self.answer_service.generate_answer(
            question=question,
            evidence=reranked_evidence,
        )

        return RAGResponse(
            question=question,
            answer=answer,
            evidence=reranked_evidence,
        )

