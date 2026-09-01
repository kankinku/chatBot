import pytest

from Server.backend.chatbot_backend.configuration_security import (
    ConfigurationError,
    validate_settings,
)


VALID_SECRET = "s" * 50


def _settings(**overrides):
    values = {
        "environment": "development",
        "secret_key": "replace-with-a-random-secret",
        "debug": True,
        "allowed_hosts": ["localhost"],
        "cors_allow_all_origins": False,
        "allow_anonymous_local": False,
        "mysql_password": "change-me",
    }
    values.update(overrides)
    return values


def test_environment_must_be_explicit_and_known():
    for environment in ("", "prod", "staging"):
        with pytest.raises(ConfigurationError):
            validate_settings(**_settings(environment=environment))


@pytest.mark.parametrize(
    "secret_key",
    [
        "",
        "django-insecure-change-this-in-production",
        "replace-with-a-random-secret",
        "chatbot-secret-key-change-in-production",
        "short-secret",
    ],
)
def test_production_rejects_placeholder_or_weak_secret(secret_key):
    with pytest.raises(ConfigurationError):
        validate_settings(
            **_settings(environment="production", secret_key=secret_key, debug=False)
        )


def test_production_accepts_explicit_strong_secret_and_database_password():
    validate_settings(
        **_settings(
            environment="production",
            secret_key=VALID_SECRET,
            debug=False,
            allowed_hosts=["chatbot.example.com"],
            mysql_password="a-real-production-password",
        )
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"debug": True},
        {"allowed_hosts": ["*"]},
        {"cors_allow_all_origins": True},
        {"allow_anonymous_local": True},
        {"mysql_password": "1234"},
        {"mysql_password": "change-me"},
    ],
)
def test_production_rejects_insecure_settings(overrides):
    values = _settings(
        environment="production",
        secret_key=VALID_SECRET,
        debug=False,
        allowed_hosts=["chatbot.example.com"],
        mysql_password="a-real-production-password",
    )
    values.update(overrides)
    with pytest.raises(ConfigurationError):
        validate_settings(**values)
