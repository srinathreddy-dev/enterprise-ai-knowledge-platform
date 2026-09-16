from pathlib import Path

import fitz
import pytest

from app.ingestion.pdf_loader import PDFLoader


def create_test_pdf(file_path: Path) -> None:
    """Create a small PDF fixture for the loader test."""
    document = fitz.open()

    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Enterprise Knowledge Platform Test Document",
    )
    page.insert_text(
        (72, 100),
        "This document is used to verify PDF text extraction.",
    )

    document.save(file_path)
    document.close()


def test_pdf_loader_extracts_page_text(tmp_path: Path):
    """The loader should extract text while preserving page numbers."""
    pdf_path = tmp_path / "test_document.pdf"

    create_test_pdf(pdf_path)

    loader = PDFLoader()
    document = loader.load(pdf_path)

    assert document.file_name == "test_document.pdf"
    assert len(document.pages) == 1
    assert document.pages[0].page_number == 1
    assert "Enterprise Knowledge Platform Test Document" in document.pages[0].text
    assert "verify PDF text extraction" in document.pages[0].text


def test_pdf_loader_rejects_missing_file(tmp_path: Path):
    """The loader should clearly report when a PDF does not exist."""
    loader = PDFLoader()

    missing_file = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        loader.load(missing_file)


def test_pdf_loader_rejects_non_pdf_file(tmp_path: Path):
    """The loader should reject files that are not PDFs."""
    loader = PDFLoader()

    text_file = tmp_path / "notes.txt"
    text_file.write_text("This is not a PDF.")

    with pytest.raises(ValueError, match="Expected a PDF file"):
        loader.load(text_file)
