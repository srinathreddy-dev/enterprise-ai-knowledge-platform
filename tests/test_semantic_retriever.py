
import numpy as np

from app.ingestion.chunker import TextChunk
from app.retrieval.semantic_retriever import SemanticEmbedder


class MockEmbedding:
    def __init__(self, embedding):
        self.embedding = embedding


class MockEmbeddingsResponse:
    def __init__(self, vectors):
        self.data = [MockEmbedding(v) for v in vectors]


class MockEmbeddingsAPI:
    def create(self, model, input):
        if isinstance(input, str):
            texts = [input]
        else:
            texts = input

        vectors = [
            [1.0, 0.0, 0.0] if "health" in text.lower()
            else [0.0, 1.0, 0.0]
            for text in texts
        ]
        return MockEmbeddingsResponse(vectors)


class MockOpenAI:
    def __init__(self):
        self.embeddings = MockEmbeddingsAPI()


def create_chunks():
    return [
        TextChunk(
            text="Employees can enroll in health insurance.",
            page_number=1,
            chunk_index=0,
        ),
        TextChunk(
            text="Employees receive paid vacation days.",
            page_number=2,
            chunk_index=1,
        ),
    ]


def test_embed_chunks_returns_one_vector_per_chunk():
    embedder = SemanticEmbedder(client=MockOpenAI())

    embeddings = embedder.embed_chunks(create_chunks())

    assert embeddings.shape == (2, 3)


def test_embed_query_returns_vector():
    embedder = SemanticEmbedder(client=MockOpenAI())

    embedding = embedder.embed_query("health insurance")

    assert embedding.shape == (3,)


def test_embeddings_are_normalized():
    embedder = SemanticEmbedder(client=MockOpenAI())

    embedding = embedder.embed_query("health insurance")

    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_empty_chunks_return_empty_matrix():
    embedder = SemanticEmbedder(client=MockOpenAI())

    embeddings = embedder.embed_chunks([])

    assert embeddings.shape == (0, 1536)
