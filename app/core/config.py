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


settings = Settings()
