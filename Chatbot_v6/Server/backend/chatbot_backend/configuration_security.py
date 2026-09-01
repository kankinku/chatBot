"""Framework-independent validation for deployment-sensitive settings."""

from typing import Iterable


class ConfigurationError(ValueError):
    """Raised when deployment settings would fail closed for security."""


_ALLOWED_ENVIRONMENTS = {"development", "production"}
_INSECURE_SECRET_KEYS = {
    "",
    "django-insecure-change-this-in-production",
    "replace-with-a-random-secret",
    "chatbot-secret-key-change-in-production",
}
_INSECURE_DATABASE_PASSWORDS = {"", "1234", "change-me", "password", "root"}
_MINIMUM_PRODUCTION_SECRET_LENGTH = 50
_MINIMUM_PRODUCTION_CREDENTIAL_UNIQUE_CHARS = 12


def _has_sufficient_credential_entropy(value: str) -> bool:
    normalized = value.strip()
    return len(set(normalized)) >= _MINIMUM_PRODUCTION_CREDENTIAL_UNIQUE_CHARS


def validate_settings(
    *,
    environment: str,
    secret_key: str,
    debug: bool,
    allowed_hosts: Iterable[str],
    cors_allow_all_origins: bool,
    allow_anonymous_local: bool,
    mysql_password: str,
) -> None:
    """Validate deployment settings and raise on unsafe production values."""

    environment = environment.strip().lower()
    if environment not in _ALLOWED_ENVIRONMENTS:
        raise ConfigurationError(
            "ENVIRONMENT must be explicitly set to development or production"
        )

    if environment != "production":
        return

    normalized_secret = secret_key.strip().lower()
    if (
        normalized_secret in _INSECURE_SECRET_KEYS
        or len(secret_key.strip()) < _MINIMUM_PRODUCTION_SECRET_LENGTH
        or not _has_sufficient_credential_entropy(secret_key)
    ):
        raise ConfigurationError(
            "A unique production SECRET_KEY of at least 50 characters is required"
        )
    if debug:
        raise ConfigurationError("DEBUG must be False in production")
    if "*" in allowed_hosts:
        raise ConfigurationError("ALLOWED_HOSTS must not contain * in production")
    if cors_allow_all_origins:
        raise ConfigurationError("CORS_ALLOW_ALL_ORIGINS must be False in production")
    if allow_anonymous_local:
        raise ConfigurationError(
            "CHATBOT_ALLOW_ANONYMOUS_LOCAL must be False in production"
        )
    if (
        mysql_password.strip().lower() in _INSECURE_DATABASE_PASSWORDS
        or not _has_sufficient_credential_entropy(mysql_password)
    ):
        raise ConfigurationError("A real MYSQL_PASSWORD is required in production")
