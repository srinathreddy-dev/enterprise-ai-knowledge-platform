from app.core.config import settings


def test_default_application_settings():
    assert settings.app_name == "Enterprise AI Knowledge Platform"
    assert settings.environment == "development"
