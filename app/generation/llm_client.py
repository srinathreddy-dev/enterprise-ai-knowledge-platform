from openai import OpenAI

from app.core.config import settings


class LLMClient:
    """Generate answers using the configured OpenAI model."""

    def __init__(self, client: OpenAI | None = None) -> None:
        if client is None:
            if not settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY must be configured to use the LLM."
                )

            client = OpenAI(api_key=settings.openai_api_key)

        self.client = client

    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
    ) -> str:
        """Generate a response from a supplied prompt."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        response = self.client.responses.create(
            model=settings.openai_model,
            input=prompt,
            max_output_tokens=max_tokens,
        )

        answer = response.output_text.strip()

        if not answer:
            raise RuntimeError("The model returned an empty response.")

        return answer
