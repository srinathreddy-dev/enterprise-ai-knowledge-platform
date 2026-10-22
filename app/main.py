
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.generation.rag_service import RAGService


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)


class AskRequest(BaseModel):
    """Request body for asking a question."""

    question: str = Field(min_length=1, max_length=2000)


class EvidenceResponse(BaseModel):
    """A piece of evidence supporting an answer."""

    page_number: int
    chunk_id: str
    text: str
    relevance_score: float


class AskResponse(BaseModel):
    """Answer and supporting evidence returned by the API."""

    question: str
    answer: str
    evidence: list[EvidenceResponse]


rag_service: RAGService | None = None


def set_rag_service(service: RAGService | None) -> None:
    """Configure the RAG service, including for tests."""
    global rag_service
    rag_service = service


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the current health status of the API."""
    return {"status": "healthy"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """Answer a question using retrieved document evidence."""
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="The knowledge service is not configured.",
        )

    try:
        result = rag_service.ask(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return AskResponse(
        question=result.question,
        answer=result.answer,
        evidence=[
            EvidenceResponse(
                page_number=item.result.chunk.page_number,
                chunk_id=item.result.chunk.chunk_id,
                text=item.result.chunk.text,
                relevance_score=item.score,
            )
            for item in result.evidence
        ],
    )

