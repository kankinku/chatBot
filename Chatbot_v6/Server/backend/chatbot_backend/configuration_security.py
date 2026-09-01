"""Framework-independent validation for deployment-sensitive settings."""

from collections import Counter
from math import log2
from string import ascii_lowercase, digits
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
_MINIMUM_PRODUCTION_DATABASE_PASSWORD_LENGTH = 20
_MINIMUM_PRODUCTION_CREDENTIAL_UNIQUE_CHARS = 12
_MINIMUM_PRODUCTION_CREDENTIAL_ENTROPY = 3.5


def _is_repeated_pattern(value: str) -> bool:
    for size in range(1, len(value) // 2 + 1):
        if len(value) % size == 0 and value == value[:size] * (len(value) // size):
            return True
    return False


def _has_repeated_motif(value: str) -> bool:
    for size in range(1, min(len(value) // 3, 32) + 1):
        motif = value[:size]
        position = 0
        repetitions = 0
        while value.startswith(motif, position):
            repetitions += 1
            position += size
        if repetitions >= 3:
            return True
    return False


def _has_sequential_run(value: str) -> bool:
    normalized = value.lower()
    alphabets = (
        digits + ascii_lowercase,
        ascii_lowercase + digits,
    )
    for alphabet in alphabets:
        for size in range(8, len(alphabet) + 1):
            if alphabet[:size] in normalized or alphabet[:size][::-1] in normalized:
                return True
    return any(
        all(
            ord(normalized[index + offset + 1])
            - ord(normalized[index + offset])
            in (1, -1)
            for offset in range(7)
        )
        for index in range(len(normalized) - 7)
    )


def _has_sufficient_credential_entropy(value: str) -> bool:
    normalized = value.strip()
    if len(set(normalized)) < _MINIMUM_PRODUCTION_CREDENTIAL_UNIQUE_CHARS:
        return False
    if _is_repeated_pattern(normalized) or _has_repeated_motif(normalized):
        return False
    if _has_sequential_run(normalized):
        return False
    frequencies = Counter(normalized)
    length = len(normalized)
    entropy = -sum(
        (count / length) * log2(count / length)
        for count in frequencies.values()
    )
    return entropy >= _MINIMUM_PRODUCTION_CREDENTIAL_ENTROPY


def validate_settings(
    *,
    environment: str,
    secret_key: str,
    debug: bool,
    allowed_hosts: Iterable[str],
    cors_allow_all_origins: bool,
    allow_anonymous_local: bool,
    mysql_password: str,
    mysql_root_password: str = "",
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
        len(mysql_password.strip()) < _MINIMUM_PRODUCTION_DATABASE_PASSWORD_LENGTH
        or mysql_password.strip().lower() in _INSECURE_DATABASE_PASSWORDS
        or not _has_sufficient_credential_entropy(mysql_password)
    ):
        raise ConfigurationError("A real MYSQL_PASSWORD is required in production")
    if (
        len(mysql_root_password.strip()) < _MINIMUM_PRODUCTION_DATABASE_PASSWORD_LENGTH
        or mysql_root_password.strip().lower() in _INSECURE_DATABASE_PASSWORDS
        or not _has_sufficient_credential_entropy(mysql_root_password)
    ):
        raise ConfigurationError("A real MYSQL_ROOT_PASSWORD is required in production")
