import pytest

from Server.backend.chatbot_backend.configuration_security import (
    ConfigurationError,
    validate_settings,
)


VALID_SECRET = "V8mQ2rL7xN4pK9dT6wC3zH5sJ1fB0uY8eA6iO4nP2rS9vX7cD5qG6hM3"


def _settings(**overrides):
    values = {
        "environment": "development",
        "secret_key": "replace-with-a-random-secret",
        "debug": True,
        "allowed_hosts": ["localhost"],
        "cors_allow_all_origins": False,
        "allow_anonymous_local": False,
        "mysql_password": "change-me",
        "mysql_root_password": VALID_SECRET,
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
            mysql_password="Q7mR2xV9kL4pN8dT6wC3zH5sJ1fB0uY8eA6iO4nP2rS9vX7cD5qG6hM3",
            mysql_root_password="H3nK8vR2xQ6mT9pL4cW7zD5sF1aB0uY8eI6oG4jN2rS9vX7cM5hP3",
        )
    )


def test_production_rejects_repeated_credentials_even_when_long_enough():
    values = _settings(
        environment="production",
        debug=False,
        allowed_hosts=["chatbot.example.com"],
        secret_key=VALID_SECRET,
        mysql_password="a" * 50,
    )
    with pytest.raises(ConfigurationError):
        validate_settings(**values)

    values["mysql_password"] = "Q7mR2xV9kL4pN8dT6wC3zH5sJ1fB0uY8eA6iO4nP2rS9vX7cD5qG6hM3"
    values["secret_key"] = "a" * 50
    with pytest.raises(ConfigurationError):
        validate_settings(**values)


def test_production_rejects_periodic_credentials_and_weak_root_password():
    values = _settings(
        environment="production",
        debug=False,
        allowed_hosts=["chatbot.example.com"],
        secret_key="Ab12Cd34Ef56" * 5,
        mysql_password="Q7mR2xV9kL4pN8dT6wC3zH5sJ1fB0uY8eA6iO4nP2rS9vX7cD5qG6hM3",
        mysql_root_password="H3nK8vR2xQ6mT9pL4cW7zD5sF1aB0uY8eI6oG4jN2rS9vX7cM5hP3",
    )
    with pytest.raises(ConfigurationError):
        validate_settings(**values)


def test_production_rejects_sequential_and_motif_suffix_credentials():
    values = _settings(
        environment="production",
        debug=False,
        allowed_hosts=["chatbot.example.com"],
        secret_key="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        mysql_password=VALID_SECRET,
        mysql_root_password=VALID_SECRET,
    )
    with pytest.raises(ConfigurationError):
        validate_settings(**values)

    values["secret_key"] = "Ab12Cd34Ef56" * 3 + "Z9x"
    with pytest.raises(ConfigurationError):
        validate_settings(**values)

    values["secret_key"] = VALID_SECRET
    values["mysql_root_password"] = "Ab12Cd34Ef56" * 5
    with pytest.raises(ConfigurationError):
        validate_settings(**values)


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
        mysql_password="Q7mR2xV9kL4pN8dT6wC3zH5sJ1fB0uY8eA6iO4nP2rS9vX7cD5qG6hM3",
    )
    values.update(overrides)
    with pytest.raises(ConfigurationError):
        validate_settings(**values)
