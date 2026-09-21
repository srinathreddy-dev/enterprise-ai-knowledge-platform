from pathlib import Path

import fitz

from app.ingestion.chunker import TextChunker
from app.ingestion.pipeline import DocumentIngestionPipeline


def create_test_pdf(file_path: Path) -> None:
    """Create a PDF containing formatting noise for the pipeline test."""
    document = fitz.open()

    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Employee    Benefits",
    )
    page.insert_text(
        (72, 100),
        "Employees are eligible for health insurance.",
    )

    document.save(file_path)
    document.close()


def test_ingestion_pipeline_cleans_and_chunks_document(tmp_path: Path):
    """The pipeline should clean text and produce searchable chunks."""
    pdf_path = tmp_path / "employee_handbook.pdf"
    create_test_pdf(pdf_path)

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

    document = fitz.open()

    for page_number in range(2):
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
