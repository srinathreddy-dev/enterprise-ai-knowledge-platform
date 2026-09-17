import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.ingestion.chunker import TextChunk


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved chunk and its relevance score."""

    chunk: TextChunk
    score: float


class BM25Retriever:
    """Retrieve document chunks using BM25 lexical matching."""

    def __init__(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")

        self.chunks = chunks

        tokenized_chunks = [
            self._tokenize(chunk.text)
            for chunk in chunks
        ]

        self.index = BM25Okapi(tokenized_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Return the highest-scoring chunks for a query."""
        if not query.strip():
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_tokens = self._tokenize(query)
        scores = self.index.get_scores(query_tokens)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        return [
            RetrievalResult(
                chunk=self.chunks[index],
                score=float(scores[index]),
            )
            for index in ranked_indexes
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Convert text into simple lowercase word tokens."""
        return re.findall(r"\b\w+\b", text.lower())
