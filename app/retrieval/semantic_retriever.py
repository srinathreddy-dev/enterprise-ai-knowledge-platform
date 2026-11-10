
from openai import OpenAI
import numpy as np

from app.core.config import settings
from app.ingestion.chunker import TextChunk


class SemanticEmbedder:
    """Generate semantic embeddings using OpenAI's hosted API."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        client: OpenAI | None = None,
    ) -> None:
        if client is None:
            if not settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY must be configured for embeddings."
                )
            client = OpenAI(api_key=settings.openai_api_key)

        self.client = client
        self.model_name = model_name

    def embed_chunks(self, chunks: list[TextChunk]):
        """Generate normalized embeddings for document chunks."""
        texts = [chunk.text for chunk in chunks]

        if not texts:
            return np.empty((0, 1536), dtype=np.float32)

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )

        vectors = [item.embedding for item in response.data]
        return self._normalize(vectors)

    def embed_query(self, query: str):
        """Generate a normalized embedding for a search query."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        response = self.client.embeddings.create(
            model=self.model_name,
            input=query,
        )

        return self._normalize([response.data[0].embedding])[0]

    @staticmethod
    def _normalize(vectors):
        """Normalize vectors to unit length."""
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)
