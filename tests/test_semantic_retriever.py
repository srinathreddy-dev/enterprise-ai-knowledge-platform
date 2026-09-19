import numpy as np

from app.ingestion.chunker import TextChunk
from app.retrieval.semantic_retriever import SemanticEmbedder


def create_chunks() -> list[TextChunk]:
    """Create representative chunks for embedding tests."""
    return [
        TextChunk(
            text="Employees can enroll in health insurance during open enrollment.",
            page_number=1,
            chunk_index=0,
        ),
        TextChunk(
            text="Employees receive fifteen paid vacation days each year.",
            page_number=2,
            chunk_index=1,
        ),
    ]


def test_embed_chunks_returns_one_vector_per_chunk():
    embedder = SemanticEmbedder()

    embeddings = embedder.embed_chunks(create_chunks())

    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0


def test_embed_query_returns_vector():
    embedder = SemanticEmbedder()

    embedding = embedder.embed_query("health insurance")

    assert embedding.ndim == 1
    assert embedding.shape[0] > 0


def test_embeddings_are_normalized():
    embedder = SemanticEmbedder()

    embedding = embedder.embed_query("health insurance")

    vector_norm = np.linalg.norm(embedding)

    assert np.isclose(vector_norm, 1.0, atol=1e-5)
