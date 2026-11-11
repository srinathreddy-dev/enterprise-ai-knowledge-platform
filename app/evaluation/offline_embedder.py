
import hashlib
import re

import numpy as np

from app.ingestion.chunker import TextChunk


class OfflineEmbedder:
    """
    Deterministic local feature-hashing embedder.

    This is an offline lexical-vector baseline, not a
    pretrained semantic embedding model.
    """

    def __init__(self, dimensions: int = 512) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")

        self.dimensions = dimensions

    def embed_chunks(
        self,
        chunks: list[TextChunk],
    ) -> np.ndarray:
        """Generate normalized vectors for document chunks."""
        if not chunks:
            return np.empty(
                (0, self.dimensions),
                dtype=np.float32,
            )

        return np.vstack([
            self._embed(chunk.text)
            for chunk in chunks
        ])

    def embed_query(self, query: str) -> np.ndarray:
        """Generate a normalized vector for a search query."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        return self._embed(query)

    def _embed(self, text: str) -> np.ndarray:
        tokens = re.findall(r"\b\w+\b", text.lower())

        features = list(tokens)

        # Add adjacent word pairs to preserve simple phrases.
        features.extend(
            f"{tokens[index]}_{tokens[index + 1]}"
            for index in range(len(tokens) - 1)
        )

        vector = np.zeros(
            self.dimensions,
            dtype=np.float32,
        )

        for feature in features:
            digest = hashlib.blake2b(
                feature.encode("utf-8"),
                digest_size=8,
            ).digest()

            value = int.from_bytes(digest, "little")

            index = value % self.dimensions
            sign = 1.0 if (value >> 8) & 1 else -1.0

            vector[index] += sign

        norm = np.linalg.norm(vector)

        if norm > 0:
            vector /= norm

        return vector
