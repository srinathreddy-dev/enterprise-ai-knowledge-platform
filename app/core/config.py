from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    app_name: str = os.getenv(
        "APP_NAME",
        "Enterprise AI Knowledge Platform",
    )
    environment: str = os.getenv(
        "ENVIRONMENT",
        "development",
    )
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv(
        "OPENAI_MODEL",
        "gpt-4.1-mini",
    )


settings = Settings()
