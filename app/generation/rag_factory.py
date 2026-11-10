
from pathlib import Path

from app.generation.answer_service import AnswerGenerationService
from app.generation.llm_client import LLMClient
from app.generation.rag_service import RAGService
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.hybrid_retriever import HybridRetriever


def build_rag_service(
    pdf_path: str | Path | None = None,
    answer_service: AnswerGenerationService | None = None,
    *,
    documents_directory: str | Path | None = None,
) -> RAGService:
    """Build a RAG service from one PDF or a directory of PDFs."""

    if pdf_path is not None and documents_directory is not None:
        raise ValueError(
            "Provide either pdf_path or documents_directory, not both."
        )

    if pdf_path is None and documents_directory is None:
        raise ValueError(
            "A PDF path or documents directory must be provided."
        )

    pipeline = DocumentIngestionPipeline()

    if documents_directory is not None:
        ingested_documents = pipeline.process_directory(
            documents_directory
        )
    else:
        source_path = Path(pdf_path)

        if source_path.is_dir():
            ingested_documents = pipeline.process_directory(source_path)
        else:
            ingested_documents = [pipeline.process(source_path)]

    all_chunks = [
        chunk
        for ingested_document in ingested_documents
        for chunk in ingested_document.chunks
    ]

    if not all_chunks:
        raise ValueError(
            "The PDF documents did not contain any searchable text."
        )

    hybrid_retriever = HybridRetriever(all_chunks)
    reranker = CrossEncoderReranker()

    if answer_service is None:
        answer_service = AnswerGenerationService(
            llm_client=LLMClient(),
        )

    return RAGService(
        chunks=all_chunks,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        answer_service=answer_service,
    )

