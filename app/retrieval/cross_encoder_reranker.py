
import re
from dataclasses import dataclass

from app.ingestion.chunker import TextChunk


@dataclass(frozen=True)
class RerankedResult:
    """A chunk scored by lightweight lexical relevance."""

    chunk: TextChunk
    score: float


class CrossEncoderReranker:
    """Rerank chunks using lightweight lexical relevance."""

    def rerank(
        self,
        query: str,
        chunks: list[TextChunk],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        if not query.strip():
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not chunks:
            return []

        query_terms = set(re.findall(r"\b\w+\b", query.lower()))
        results = []

        for chunk in chunks:
            document_terms = set(
                re.findall(r"\b\w+\b", chunk.text.lower())
            )

            overlap = len(query_terms & document_terms)
            score = overlap / max(len(query_terms), 1)

            results.append(
                RerankedResult(
                    chunk=chunk,
                    score=float(score),
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]
