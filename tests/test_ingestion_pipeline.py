from pathlib import Path

import fitz

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


def test_ingestion_pipeline_cleans_extracted_text(tmp_path: Path):
    """The pipeline should return cleaned text with document metadata."""
    pdf_path = tmp_path / "employee_handbook.pdf"
    create_test_pdf(pdf_path)

    pipeline = DocumentIngestionPipeline()
    document = pipeline.process(pdf_path)

    assert document.file_name == "employee_handbook.pdf"
    assert len(document.pages) == 1
    assert document.pages[0].page_number == 1
    assert "Employee Benefits" in document.pages[0].text
    assert "Employees are eligible for health insurance." in document.pages[0].text
