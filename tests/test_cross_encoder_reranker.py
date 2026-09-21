import pytest

from app.ingestion.chunker import TextChunk
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.bm25_retriever import RetrievalResult


def create_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk=TextChunk(
                text="Employees receive fifteen paid vacation days each year.",
                page_number=1,
                chunk_index=0,
            ),
            score=0.5,
        ),
        RetrievalResult(
            chunk=TextChunk(
                text="The company provides retirement savings plans.",
                page_number=2,
                chunk_index=0,
            ),
            score=0.4,
        ),
    ]


@pytest.fixture(scope="module")
def reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


def test_reranker_prioritizes_relevant_result(
    reranker: CrossEncoderReranker,
):
    results = reranker.rerank(
        "How many vacation days do employees receive?",
        create_results(),
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].result.chunk.page_number == 1
    assert results[0].score > results[1].score


def test_reranker_respects_top_k(
    reranker: CrossEncoderReranker,
):
    results = reranker.rerank(
        "How many vacation days do employees receive?",
        create_results(),
        top_k=1,
    )

    assert len(results) == 1


def test_reranker_returns_empty_for_empty_query(
    reranker: CrossEncoderReranker,
):
    results = reranker.rerank(
        "   ",
        create_results(),
    )

    assert results == []


def test_reranker_returns_empty_for_empty_results(
    reranker: CrossEncoderReranker,
):
    results = reranker.rerank(
        "How many vacation days do employees receive?",
        [],
    )

    assert results == []


def test_reranker_rejects_invalid_top_k(
    reranker: CrossEncoderReranker,
):
    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank(
            "How many vacation days do employees receive?",
            create_results(),
            top_k=0,
        )
