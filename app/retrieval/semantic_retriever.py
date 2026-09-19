from sentence_transformers import SentenceTransformer

from app.ingestion.chunker import TextChunk


class SemanticEmbedder:
    """Generate semantic embeddings for document chunks and queries."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: list[TextChunk]):
        """Generate embeddings for document chunks."""
        texts = [chunk.text for chunk in chunks]

        return self.model.encode(
            texts,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str):
        """Generate an embedding for a search query."""
        return self.model.encode(
            query,
            normalize_embeddings=True,
        )
