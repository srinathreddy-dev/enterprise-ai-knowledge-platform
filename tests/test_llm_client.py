from unittest.mock import Mock

import pytest

from app.generation.llm_client import LLMClient


def create_mock_client(answer: str = "Employees receive 15 vacation days.") -> Mock:
    """Create a mock OpenAI client with a predictable response."""
    client = Mock()
    client.responses.create.return_value.output_text = answer
    return client


def test_generate_returns_model_answer():
    client = create_mock_client()
    llm_client = LLMClient(client=client)

    answer = llm_client.generate("How many vacation days are provided?")

    assert answer == "Employees receive 15 vacation days."
    client.responses.create.assert_called_once()


def test_generate_strips_whitespace():
    client = create_mock_client("  Answer with whitespace.  ")
    llm_client = LLMClient(client=client)

    answer = llm_client.generate("Provide an answer.")

    assert answer == "Answer with whitespace."


def test_generate_rejects_empty_prompt():
    llm_client = LLMClient(client=create_mock_client())

    with pytest.raises(ValueError, match="Prompt"):
        llm_client.generate("   ")


def test_generate_rejects_invalid_max_tokens():
    llm_client = LLMClient(client=create_mock_client())

    with pytest.raises(ValueError, match="max_tokens"):
        llm_client.generate("Provide an answer.", max_tokens=0)


def test_generate_rejects_empty_model_response():
    llm_client = LLMClient(client=create_mock_client("   "))

    with pytest.raises(RuntimeError, match="empty response"):
        llm_client.generate("Provide an answer.")
