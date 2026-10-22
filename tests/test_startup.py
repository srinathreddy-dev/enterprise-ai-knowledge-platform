from fastapi.testclient import TestClient
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import FastAPI

from app.main import lifespan


def test_startup_records_error_when_sample_pdf_is_missing():
    test_app = FastAPI(lifespan=lifespan)

    fake_settings = Mock()
    fake_settings.openai_api_key = "test-key"

    with (
        patch("app.main.SAMPLE_PDF", Path("missing-handbook.pdf")),
        patch("app.main.settings", fake_settings),
        patch("app.main.build_rag_service") as mock_builder,
    ):
        with TestClient(test_app):
            assert test_app.state.rag_service is None
            assert "Sample PDF not found" in (
                test_app.state.rag_startup_error
            )
            mock_builder.assert_not_called()


def test_startup_skips_model_loading_when_api_key_is_missing():
    test_app = FastAPI(lifespan=lifespan)

    fake_settings = Mock()
    fake_settings.openai_api_key = None

    with (
        patch(
            "app.main.SAMPLE_PDF",
            Path("data/sample/employee_handbook.pdf"),
        ),
        patch("app.main.settings", fake_settings),
        patch("app.main.build_rag_service") as mock_builder,
    ):
        with TestClient(test_app):
            assert test_app.state.rag_service is None
            assert test_app.state.rag_startup_error == (
                "OpenAI API key is not configured."
            )
            mock_builder.assert_not_called()

