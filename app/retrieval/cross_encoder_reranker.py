
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from app.ingestion.chunker import TextChunk


@dataclass(frozen=True)
class RerankedResult:
    """A chunk scored by a cross-encoder."""

    chunk: TextChunk
    score: float


class CrossEncoderReranker:
    """Rerank retrieved chunks using query-document relevance."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list[TextChunk],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        """Return candidate chunks sorted by cross-encoder score."""

        if not query.strip():
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not chunks:
            return []

        pairs = [
            (query, chunk.text)
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        results = [
            RerankedResult(
                chunk=chunk,
                score=float(score),
            )
            for chunk, score in zip(chunks, scores)
        ]

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]
