import pytest
from unittest.mock import Mock

from app.generation.answer_service import AnswerGenerationService
from app.ingestion.chunker import TextChunk
from app.reranking.cross_encoder_reranker import RerankedResult
from app.retrieval.bm25_retriever import RetrievalResult


def create_evidence() -> list[RerankedResult]:
    """Create representative reranked evidence."""
    chunk = TextChunk(
        text="Employees receive fifteen paid vacation days each year.",
        page_number=2,
        chunk_index=0,
    )

    return [
        RerankedResult(
            result=RetrievalResult(chunk=chunk, score=0.8),
            score=9.5,
        )
    ]


def test_service_builds_prompt_and_returns_generated_answer():
    prompt_builder = Mock()
    prompt_builder.build.return_value = (
        "Answer using the supplied evidence."
    )

    llm_client = Mock()
    llm_client.generate.return_value = (
        "Employees receive fifteen paid vacation days each year. [Page 2]"
    )

    service = AnswerGenerationService(
        prompt_builder=prompt_builder,
        llm_client=llm_client,
    )

    answer = service.generate_answer(
        "How many vacation days do employees receive?",
        create_evidence(),
    )

    assert "fifteen paid vacation days" in answer
    assert "[Page 2]" in answer

    prompt_builder.build.assert_called_once()
    llm_client.generate.assert_called_once_with(
        "Answer using the supplied evidence."
    )


def test_service_rejects_missing_llm_client():
    service = AnswerGenerationService()

    with pytest.raises(ValueError, match="LLMClient"):
        service.generate_answer(
            "What is the vacation policy?",
            create_evidence(),
        )


def test_service_propagates_prompt_validation_errors():
    prompt_builder = Mock()
    prompt_builder.build.side_effect = ValueError(
        "Question cannot be empty"
    )

    llm_client = Mock()

    service = AnswerGenerationService(
        prompt_builder=prompt_builder,
        llm_client=llm_client,
    )

    with pytest.raises(ValueError, match="Question"):
        service.generate_answer("   ", create_evidence())

    llm_client.generate.assert_not_called()
