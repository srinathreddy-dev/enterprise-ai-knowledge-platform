import numpy as np
import pytest


class TestSemanticEmbedder:
    """Deterministic local embedder for tests; never calls OpenAI."""

    CATEGORIES = {
        "health": {
            "health", "insurance", "medical", "coverage", "enrollment"
        },
        "vacation": {
            "vacation", "days", "leave", "holiday", "time"
        },
        "retirement": {
            "retirement", "savings", "pension"
        },
    }

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def _embed(cls, text):
        terms = set(text.lower().split())

        vector = np.array(
            [
                float(bool(terms & cls.CATEGORIES["health"])),
                float(bool(terms & cls.CATEGORIES["vacation"])),
                float(bool(terms & cls.CATEGORIES["retirement"])),
            ],
            dtype=np.float32,
        )

        if np.linalg.norm(vector) == 0:
            vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        return vector / np.linalg.norm(vector)

    def embed_chunks(self, chunks):
        return np.vstack([
            self._embed(chunk.text)
            for chunk in chunks
        ])

    def embed_query(self, query):
        return self._embed(query)


@pytest.fixture(autouse=True)
def use_test_embedder(monkeypatch):
    """Prevent tests from making real OpenAI embedding requests."""

    monkeypatch.setattr(
        "app.retrieval.semantic_search.SemanticEmbedder",
        TestSemanticEmbedder,
    )
