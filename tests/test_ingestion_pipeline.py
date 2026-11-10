
from pathlib import Path

import pymupdf
import pytest

from app.ingestion.chunker import TextChunker
from app.ingestion.pipeline import DocumentIngestionPipeline


def create_test_pdf(file_path: Path, text: str = "Employee Benefits") -> None:
    """Create a PDF containing sample text."""
    document = pymupdf.open()

    page = document.new_page()
    page.insert_text((72, 72), text)

    document.save(file_path)
    document.close()


def test_ingestion_pipeline_cleans_and_chunks_document(tmp_path: Path):
    """The pipeline should clean text and produce searchable chunks."""
    pdf_path = tmp_path / "employee_handbook.pdf"

    document = pymupdf.open()
    page = document.new_page()

    page.insert_text((72, 72), "Employee    Benefits")
    page.insert_text(
        (72, 100),
        "Employees are eligible for health insurance.",
    )

    document.save(pdf_path)
    document.close()

    pipeline = DocumentIngestionPipeline()
    result = pipeline.process(pdf_path)

    assert result.document.file_name == "employee_handbook.pdf"
    assert len(result.document.pages) == 1
    assert result.document.pages[0].page_number == 1

    assert "Employee Benefits" in result.document.pages[0].text
    assert "Employees are eligible for health insurance." in (
        result.document.pages[0].text
    )

    assert len(result.chunks) == 1
    assert result.chunks[0].page_number == 1
    assert result.chunks[0].chunk_index == 0
    assert "Employee Benefits" in result.chunks[0].text


def test_ingestion_pipeline_assigns_unique_chunk_ids_across_pages(
    tmp_path: Path,
):
    """Chunks from different pages should have unique identifiers."""
    pdf_path = tmp_path / "multi_page_handbook.pdf"

    document = pymupdf.open()

    for _ in range(2):
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Employee benefits information. " * 30,
        )

    document.save(pdf_path)
    document.close()

    pipeline = DocumentIngestionPipeline(
        text_chunker=TextChunker(chunk_size=100, overlap=10),
    )

    result = pipeline.process(pdf_path)

    chunk_ids = [chunk.chunk_id for chunk in result.chunks]

    assert len(chunk_ids) > 2
    assert len(chunk_ids) == len(set(chunk_ids))


def test_ingestion_pipeline_processes_multiple_pdfs(tmp_path: Path):
    """The pipeline should process every PDF in a directory."""
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    create_test_pdf(
        documents_dir / "employee_handbook.pdf",
        "Employees receive fifteen vacation days.",
    )

    create_test_pdf(
        documents_dir / "it_security_policy.pdf",
        "Employees must protect company passwords.",
    )

    pipeline = DocumentIngestionPipeline()

    results = pipeline.process_directory(documents_dir)

    assert len(results) == 2

    file_names = [
        result.document.file_name
        for result in results
    ]

    assert file_names == [
        "employee_handbook.pdf",
        "it_security_policy.pdf",
    ]

    all_chunks = [
        chunk
        for result in results
        for chunk in result.chunks
    ]

    chunk_ids = [chunk.chunk_id for chunk in all_chunks]

    assert len(chunk_ids) == 2
    assert len(chunk_ids) == len(set(chunk_ids))

    assert "employee_handbook-page-1-chunk-0" in chunk_ids
    assert "it_security_policy-page-1-chunk-0" in chunk_ids


def test_ingestion_pipeline_rejects_empty_directory(tmp_path: Path):
    """The pipeline should reject directories without PDF files."""
    documents_dir = tmp_path / "empty_documents"
    documents_dir.mkdir()

    pipeline = DocumentIngestionPipeline()

    with pytest.raises(ValueError, match="No PDF documents found"):
        pipeline.process_directory(documents_dir)


def test_ingestion_pipeline_rejects_missing_directory(tmp_path: Path):
    """The pipeline should reject directories that do not exist."""
    missing_dir = tmp_path / "missing_documents"

    pipeline = DocumentIngestionPipeline()

    with pytest.raises(FileNotFoundError):
        pipeline.process_directory(missing_dir)


def test_ingestion_pipeline_ignores_non_pdf_files(tmp_path: Path):
    """The pipeline should only process PDF files."""
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    create_test_pdf(documents_dir / "employee_handbook.pdf")

    (documents_dir / "notes.txt").write_text(
        "This is not a PDF document."
    )

    pipeline = DocumentIngestionPipeline()

    results = pipeline.process_directory(documents_dir)

    assert len(results) == 1
    assert results[0].document.file_name == "employee_handbook.pdf"
