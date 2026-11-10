from pathlib import Path

import pymupdf
import pytest

from app.generation.rag_factory import build_rag_service


class FakeAnswerService:
    """Return a predictable answer without calling an LLM."""

    def generate_answer(self, question, evidence):
        page = evidence[0].result.chunk.page_number
        return f"Test answer based on page {page}. [Page {page}]"


def create_test_pdf(file_path: Path, text: str):
    """Create a small PDF containing searchable text."""

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(file_path)
    document.close()


def test_factory_builds_working_rag_service_from_sample_pdf():
    pdf_path = Path("data/sample/employee_handbook.pdf")

    if not pdf_path.exists():
        raise AssertionError(f"Sample PDF not found: {pdf_path}")

    service = build_rag_service(
        pdf_path=pdf_path,
        answer_service=FakeAnswerService(),
    )

    response = service.ask(
        "How many vacation days do employees receive?"
    )

    assert response.question == (
        "How many vacation days do employees receive?"
    )
    assert response.evidence
    assert response.evidence[0].result.chunk.page_number == 1
    assert "15 days of paid vacation" in (
        response.evidence[0].result.chunk.text
    )
    assert response.answer == "Test answer based on page 1. [Page 1]"


def test_factory_builds_rag_service_from_multiple_pdfs(tmp_path):
    create_test_pdf(
        tmp_path / "employee_benefits.pdf",
        "Employees receive health insurance and dental insurance.",
    )
    create_test_pdf(
        tmp_path / "it_security_policy.pdf",
        "Employees must use multi-factor authentication for company systems.",
    )

    service = build_rag_service(
        documents_directory=tmp_path,
        answer_service=FakeAnswerService(),
    )

    document_ids = {
        chunk.document_id for chunk in service.chunks
    }

    assert document_ids == {
        "employee_benefits",
        "it_security_policy",
    }
    assert len(service.chunks) == 2


def test_factory_rejects_both_pdf_and_directory_arguments(tmp_path):
    with pytest.raises(ValueError, match="either pdf_path"):
        build_rag_service(
            pdf_path="data/sample/employee_handbook.pdf",
            documents_directory=tmp_path,
            answer_service=FakeAnswerService(),
        )


def test_factory_rejects_missing_source_argument():
    with pytest.raises(ValueError, match="must be provided"):
        build_rag_service(
            answer_service=FakeAnswerService(),
        )
