
from pathlib import Path

from app.generation.rag_factory import build_rag_service


class FakeAnswerService:
    """Return a predictable answer without calling an LLM."""

    def generate_answer(self, question, evidence):
        page = evidence[0].result.chunk.page_number
        return f"Test answer based on page {page}. [Page {page}]"


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

