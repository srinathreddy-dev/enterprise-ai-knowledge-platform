from dataclasses import dataclass

import numpy as np

from app.ingestion.chunker import TextChunk
from app.retrieval.semantic_retriever import SemanticEmbedder


@dataclass(frozen=True)
class SemanticSearchResult:
    """A semantically matched chunk and its similarity score."""

    chunk: TextChunk
    score: float


class SemanticSearcher:
    """Search document chunks using cosine similarity."""

    def __init__(
        self,
        chunks: list[TextChunk],
        embedder: SemanticEmbedder | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")

        self.chunks = chunks
        self.embedder = embedder or SemanticEmbedder()
        self.embeddings = self.embedder.embed_chunks(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SemanticSearchResult]:
        """Return the chunks most similar to the query."""
        if not query.strip():
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = self.embedder.embed_query(query)

        scores = np.dot(
            self.embeddings,
            query_embedding,
        )

        ranked_indexes = np.argsort(scores)[::-1][:top_k]

        return [
            SemanticSearchResult(
                chunk=self.chunks[index],
                score=float(scores[index]),
            )
            for index in ranked_indexes
        ]
