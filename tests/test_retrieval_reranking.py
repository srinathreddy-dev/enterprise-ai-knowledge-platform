from app.ingestion.chunker import TextChunk
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.bm25_retriever import RetrievalResult
from app.retrieval.hybrid_retriever import HybridRetriever


def create_chunks() -> list[TextChunk]:
    """Create representative chunks for retrieval integration tests."""
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
        TextChunk(
            text="The company provides retirement savings plans to eligible employees.",
            page_number=3,
            chunk_index=2,
        ),
    ]


def test_hybrid_results_can_be_reranked():
    """Hybrid retrieval results should be usable by the cross-encoder."""
    chunks = create_chunks()

    retriever = HybridRetriever(chunks)

    retrieved_results = retriever.search(
        "How many vacation days do employees receive?",
        top_k=3,
    )

    reranker = CrossEncoderReranker()

    reranked_results = reranker.rerank(
        "How many vacation days do employees receive?",
        [
            RetrievalResult(
                chunk=result.chunk,
                score=result.score,
            )
            for result in retrieved_results
        ],
        top_k=2,
    )

    assert len(reranked_results) == 2
    assert reranked_results[0].result.chunk.page_number == 2
    assert (
        reranked_results[0].score
        > reranked_results[1].score
    )
