
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import lifespan


def test_startup_records_error_when_documents_directory_is_missing():
    test_app = FastAPI(lifespan=lifespan)

    fake_settings = Mock()
    fake_settings.openai_api_key = "test-key"

    with (
        patch(
            "app.main.DOCUMENTS_DIRECTORY",
            Path("missing-documents"),
        ),
        patch("app.main.settings", fake_settings),
        patch("app.main.build_rag_service") as mock_builder,
    ):
        with TestClient(test_app):
            assert test_app.state.rag_service is None
            assert "Documents directory not found" in (
                test_app.state.rag_startup_error
            )
            mock_builder.assert_not_called()


def test_startup_skips_model_loading_when_api_key_is_missing():
    test_app = FastAPI(lifespan=lifespan)

    fake_settings = Mock()
    fake_settings.openai_api_key = None

    with (
        patch(
            "app.main.DOCUMENTS_DIRECTORY",
            Path("data/sample"),
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


def test_startup_builds_knowledge_service_from_documents_directory(
    tmp_path,
):
    test_app = FastAPI(lifespan=lifespan)

    fake_settings = Mock()
    fake_settings.openai_api_key = "test-key"

    mock_service = Mock()

    with (
        patch(
            "app.main.DOCUMENTS_DIRECTORY",
            tmp_path,
        ),
        patch("app.main.settings", fake_settings),
        patch(
            "app.main.build_rag_service",
            return_value=mock_service,
        ) as mock_builder,
    ):
        with TestClient(test_app):
            assert test_app.state.rag_service is mock_service
            assert test_app.state.rag_startup_error is None

            mock_builder.assert_called_once_with(
                documents_directory=tmp_path,
            )

