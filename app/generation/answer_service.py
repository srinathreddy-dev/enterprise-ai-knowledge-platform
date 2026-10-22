from app.generation.llm_client import LLMClient
from app.generation.prompt_builder import PromptBuilder
from app.reranking.cross_encoder_reranker import RerankedResult


class AnswerGenerationService:
    """Generate answers grounded in reranked document evidence."""

    def __init__(
        self,
        prompt_builder: PromptBuilder | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.llm_client = llm_client

    def generate_answer(
        self,
        question: str,
        evidence: list[RerankedResult],
    ) -> str:
        """Build a grounded prompt and generate an answer."""
        prompt = self.prompt_builder.build(
            question=question,
            evidence=evidence,
        )

        if self.llm_client is None:
            raise ValueError(
                "An LLMClient must be provided to generate answers."
            )

        return self.llm_client.generate(prompt)
