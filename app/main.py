
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.generation.rag_factory import build_rag_service
from app.generation.rag_service import RAGService


SAMPLE_PDF = Path("data/sample/employee_handbook.pdf")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the sample knowledge base when the API starts."""
    app.state.rag_service = None
    app.state.rag_startup_error = None

    if not SAMPLE_PDF.exists():
        app.state.rag_startup_error = (
            f"Sample PDF not found: {SAMPLE_PDF}"
        )
        yield
        return

    try:
        app.state.rag_service = build_rag_service(SAMPLE_PDF)
    except (ValueError, FileNotFoundError) as exc:
        app.state.rag_startup_error = str(exc)

    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
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


def set_rag_service(service: RAGService | None) -> None:
    """Configure the RAG service, including for tests."""
    app.state.rag_service = service


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the current health status of the API."""
    return {"status": "healthy"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """Answer a question using retrieved document evidence."""
    service = getattr(app.state, "rag_service", None)

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="The knowledge service is not configured.",
        )

    try:
        result = service.ask(request.question)
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

