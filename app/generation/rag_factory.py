
from pathlib import Path

from app.generation.answer_service import AnswerGenerationService
from app.generation.llm_client import LLMClient
from app.generation.rag_service import RAGService
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.hybrid_retriever import HybridRetriever


def build_rag_service(
    pdf_path: str | Path,
    answer_service: AnswerGenerationService | None = None,
) -> RAGService:
    """Build a RAG service from a PDF and an answer-generation service."""
    ingested = DocumentIngestionPipeline().process(pdf_path)

    if not ingested.chunks:
        raise ValueError("The PDF did not contain any searchable text.")

    hybrid_retriever = HybridRetriever(ingested.chunks)
    reranker = CrossEncoderReranker()

    if answer_service is None:
        answer_service = AnswerGenerationService(
            llm_client=LLMClient(),
        )

    return RAGService(
        chunks=ingested.chunks,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        answer_service=answer_service,
    )

