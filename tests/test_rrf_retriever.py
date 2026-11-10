
import pytest

from app.ingestion.chunker import TextChunk
from app.retrieval.rrf_retriever import RRFHybridRetriever


def create_chunks() -> list[TextChunk]:
    return [
        TextChunk(
            text="Employees can enroll in health insurance during open enrollment.",
            page_number=1,
            chunk_index=0,
            document_id="employee_handbook",
        ),
        TextChunk(
            text="Employees receive fifteen paid vacation days each year.",
            page_number=2,
            chunk_index=1,
            document_id="employee_handbook",
        ),
        TextChunk(
            text="The company provides retirement savings plans to eligible employees.",
            page_number=3,
            chunk_index=2,
            document_id="employee_handbook",
        ),
    ]


def test_rrf_retriever_returns_relevant_result():
    retriever = RRFHybridRetriever(create_chunks())

    results = retriever.search(
        "How do employees enroll in medical coverage?",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.page_number == 1


def test_rrf_results_are_sorted_by_score():
    retriever = RRFHybridRetriever(create_chunks())

    results = retriever.search(
        "employee vacation time",
        top_k=3,
    )

    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)


def test_rrf_result_contains_both_ranks():
    retriever = RRFHybridRetriever(create_chunks())

    results = retriever.search(
        "health insurance",
        top_k=3,
    )

    assert len(results) == 3

    for result in results:
        assert result.bm25_rank is not None
        assert result.semantic_rank is not None
        assert result.score > 0


def test_rrf_returns_empty_list_for_empty_query():
    retriever = RRFHybridRetriever(create_chunks())

    assert retriever.search("   ") == []


def test_rrf_rejects_invalid_top_k():
    retriever = RRFHybridRetriever(create_chunks())

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("health insurance", top_k=0)


def test_rrf_rejects_invalid_rrf_constant():
    with pytest.raises(ValueError, match="rrf_k"):
        RRFHybridRetriever(create_chunks(), rrf_k=0)


def test_rrf_handles_duplicate_chunk_indexes_across_pages():
    chunks = [
        TextChunk(
            text="Health insurance enrollment is available during open enrollment.",
            page_number=1,
            chunk_index=0,
            document_id="employee_handbook",
        ),
        TextChunk(
            text="Retirement savings plans are available to eligible employees.",
            page_number=2,
            chunk_index=0,
            document_id="employee_handbook",
        ),
    ]

    retriever = RRFHybridRetriever(chunks)

    results = retriever.search(
        "retirement savings benefits",
        top_k=2,
    )

    returned_ids = {
        result.chunk.chunk_id
        for result in results
   }

    assert len(results) == 2
    assert returned_ids == {
        "employee_handbook-page-1-chunk-0",
        "employee_handbook-page-2-chunk-0",
    }
