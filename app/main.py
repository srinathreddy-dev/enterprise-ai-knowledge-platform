import logging
import secrets
import time

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import Response

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging_config import configure_logging
from app.core.metrics import (
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
)
from app.generation.rag_factory import build_rag_service
from app.generation.rag_service import RAGService


configure_logging()

logger = logging.getLogger(__name__)

DOCUMENTS_DIRECTORY = Path("data/sample")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the knowledge base when the API starts."""

    app.state.rag_service = None
    app.state.rag_startup_error = None

    if not DOCUMENTS_DIRECTORY.exists():
        app.state.rag_startup_error = (
            f"Documents directory not found: {DOCUMENTS_DIRECTORY}"
        )
        logger.error(app.state.rag_startup_error)
        yield
        return

    if not settings.openai_api_key:
        app.state.rag_startup_error = (
            "OpenAI API key is not configured."
        )
        logger.error(app.state.rag_startup_error)
        yield
        return

    try:
        app.state.rag_service = build_rag_service(
            documents_directory=DOCUMENTS_DIRECTORY,
        )
        logger.info("RAG service initialized successfully.")

    except (ValueError, FileNotFoundError):
        logger.exception("Failed to initialize RAG service.")

    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next,
):
    """Log HTTP requests and record Prometheus metrics."""

    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code

        duration_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        logger.info(
            "HTTP request completed",
            extra={
                "http_method": request.method,
                "http_path": request.url.path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
        )

        return response

    except Exception:
        duration_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        logger.exception(
            "HTTP request failed",
            extra={
                "http_method": request.method,
                "http_path": request.url.path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
        )

        raise

    finally:
        duration_seconds = time.perf_counter() - start_time

        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")

        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()

        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration_seconds)


class AskRequest(BaseModel):
    """Request body for asking a question."""

    question: str = Field(
        min_length=1,
        max_length=2000,
    )


class EvidenceResponse(BaseModel):
    """A piece of evidence supporting an answer."""

    document_name: str
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


def verify_api_key(api_key: str | None) -> None:
    """Verify the application API key."""

    expected_key = settings.app_api_key

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    if not api_key or not secrets.compare_digest(
        api_key,
        expected_key,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the current health status of the API."""

    return {"status": "healthy"}


@app.get("/ready")
def readiness_check() -> dict[str, str]:
    """Return whether the knowledge service is configured."""

    service = getattr(app.state, "rag_service", None)

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="The knowledge service is not configured.",
        )

    return {"status": "ready"}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Expose Prometheus application metrics."""

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
    ),
) -> AskResponse:
    """Answer a question using retrieved document evidence."""

    # Authenticate before accessing the RAG service.
    verify_api_key(api_key)

    service = getattr(app.state, "rag_service", None)

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="The knowledge service is not configured.",
        )

    try:
        result = service.ask(request.question)

    except ValueError as exc:
        logger.exception("Invalid request while processing /ask.")

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except RateLimitError as exc:
        logger.exception("OpenAI rate limit or quota error.")

        raise HTTPException(
            status_code=503,
            detail=(
                "The AI service is temporarily unavailable. "
                "Please try again later."
            ),
        ) from exc

    except APITimeoutError as exc:
        logger.exception("OpenAI request timed out.")

        raise HTTPException(
            status_code=504,
            detail=(
                "The AI service request timed out. "
                "Please try again."
            ),
        ) from exc

    except APIConnectionError as exc:
        logger.exception("Could not connect to OpenAI.")

        raise HTTPException(
            status_code=503,
            detail="The AI service is temporarily unreachable.",
        ) from exc

    except AuthenticationError as exc:
        logger.exception("OpenAI authentication failed.")

        raise HTTPException(
            status_code=502,
            detail=(
                "The AI service authentication is not "
                "configured correctly."
            ),
        ) from exc

    except APIStatusError as exc:
        logger.exception("OpenAI returned an API error.")

        raise HTTPException(
            status_code=502,
            detail=(
                "The AI service returned an error. "
                "Please try again later."
            ),
        ) from exc

    return AskResponse(
        question=result.question,
        answer=result.answer,
        evidence=[
            EvidenceResponse(
                document_name=(
                    f"{item.result.chunk.document_id}.pdf"
                ),
                page_number=item.result.chunk.page_number,
                chunk_id=item.result.chunk.chunk_id,
                text=item.result.chunk.text,
                relevance_score=item.score,
            )
            for item in result.evidence
        ],
    )
