import ast
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = REPO_ROOT / "Server" / "backend" / "chatbot_backend" / "settings.py"
ENV_EXAMPLE_PATH = REPO_ROOT / "Server" / "backend" / "env.example"
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
VALID_DATABASE_PASSWORD = "Q7mR2xV9kL4pN8dT6wC3zH5sJ1fB0uY8eA6iO4nP2rS9vX7cD5qG6hM3"
CONFIGURATION_SECURITY_PATH = (
    REPO_ROOT / "Server" / "backend" / "chatbot_backend" / "configuration_security.py"
)
WAIT_FOR_MYSQL_PATH = REPO_ROOT / "Server" / "backend" / "wait-for-mysql.py"


def _run_settings_import(values=None, unset=()):
    environment = {
        name: os.environ[name]
        for name in ("PATH", "SystemRoot", "WINDIR", "TEMP", "TMP")
        if name in os.environ
    }
    environment.update(values or {})
    for name in unset:
        environment.pop(name, None)
    with tempfile.TemporaryDirectory() as isolated_cwd:
        isolated_root = Path(isolated_cwd)
        shutil.copytree(
            REPO_ROOT / "Server" / "backend" / "chatbot_backend",
            isolated_root / "chatbot_backend",
        )
        environment["PYTHONPATH"] = str(isolated_root)
        return subprocess.run(
            [sys.executable, "-c", "import chatbot_backend.settings"],
            cwd=isolated_cwd,
            env=environment,
            capture_output=True,
            text=True,
        )


def _assignment(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return node.value
    raise AssertionError(f"assignment not found: {name}")


def _is_config_call(node):
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "config"
        for child in ast.walk(node)
    )


def test_security_sensitive_settings_are_environment_driven():
    tree = ast.parse(SETTINGS_PATH.read_text(encoding="utf-8"))

    assert _is_config_call(_assignment(tree, "SECRET_KEY"))
    assert _is_config_call(_assignment(tree, "DEBUG"))
    assert _is_config_call(_assignment(tree, "ALLOWED_HOSTS"))
    assert _is_config_call(_assignment(tree, "CORS_ALLOW_ALL_ORIGINS"))
    assert _is_config_call(_assignment(tree, "CHATBOT_ALLOW_ANONYMOUS_LOCAL"))


def test_security_defaults_do_not_enable_wildcard_or_anonymous_access():
    source = SETTINGS_PATH.read_text(encoding="utf-8")
    assert "ALLOWED_HOSTS = ['*']" not in source
    assert "CORS_ALLOW_ALL_ORIGINS = True" not in source
    assert "CHATBOT_ALLOW_ANONYMOUS_LOCAL = True" not in source
    assert "default='1234'" not in source
    assert "default='development'" not in source
    assert "validate_settings(" in source


def test_known_deployment_placeholders_are_validated_centrally():
    source = CONFIGURATION_SECURITY_PATH.read_text(encoding="utf-8")
    assert "replace-with-a-random-secret" in source
    assert "chatbot-secret-key-change-in-production" in source
    assert "1234" in source
    assert "change-me" in source
    assert "_MINIMUM_PRODUCTION_SECRET_LENGTH = 50" in source


def test_env_example_uses_safe_explicit_development_values():
    source = ENV_EXAMPLE_PATH.read_text(encoding="utf-8")
    assert "ENVIRONMENT=development" in source
    assert "<SET_IN_SECRET_MANAGER>" in source
    assert "ALLOWED_HOSTS=localhost,127.0.0.1" in source
    assert "CORS_ALLOW_ALL_ORIGINS=False" in source
    assert "CHATBOT_ALLOW_ANONYMOUS_LOCAL=True" in source
    assert "MYSQL_PASSWORD=1234" not in source


def test_docker_compose_declares_its_nonproduction_environment():
    source = COMPOSE_PATH.read_text(encoding="utf-8")
    assert "- ENVIRONMENT=${ENVIRONMENT:-development}" in source


def test_mysql_wait_script_uses_required_runtime_credentials():
    source = WAIT_FOR_MYSQL_PATH.read_text(encoding="utf-8")
    for name in ("MYSQL_HOST", "MYSQL_PORT", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"):
        assert f"required_env('{name}')" in source
    assert "password='1234'" not in source
    assert "user='chatbot_user'" not in source


def test_settings_import_fails_closed_when_environment_is_missing():
    result = _run_settings_import(
        unset=(
            "ENVIRONMENT",
        )
    )

    assert result.returncode != 0
    assert "ENVIRONMENT must be explicitly set" in result.stderr


def test_settings_import_fails_closed_when_production_secret_is_missing():
    result = _run_settings_import(
        {
            "ENVIRONMENT": "production",
            "DEBUG": "False",
            "ALLOWED_HOSTS": "chatbot.example.com",
            "CORS_ALLOW_ALL_ORIGINS": "False",
            "CHATBOT_ALLOW_ANONYMOUS_LOCAL": "False",
            "MYSQL_PASSWORD": VALID_DATABASE_PASSWORD,
            "MYSQL_ROOT_PASSWORD": VALID_DATABASE_PASSWORD,
        },
        unset=("SECRET_KEY",),
    )

    assert result.returncode != 0
    assert "SECRET_KEY" in result.stderr


def test_settings_import_fails_closed_when_production_database_password_is_missing():
    result = _run_settings_import(
        {
            "ENVIRONMENT": "production",
            "SECRET_KEY": "V8mQ2rL7xN4pK9dT6wC3zH5sJ1fB0uY8eA6iO4nP2rS9vX7cD5qG6hM3",
            "DEBUG": "False",
            "ALLOWED_HOSTS": "chatbot.example.com",
            "CORS_ALLOW_ALL_ORIGINS": "False",
            "CHATBOT_ALLOW_ANONYMOUS_LOCAL": "False",
            "MYSQL_ROOT_PASSWORD": VALID_DATABASE_PASSWORD,
        },
        unset=("MYSQL_PASSWORD",),
    )

    assert result.returncode != 0
    assert "MYSQL_PASSWORD" in result.stderr


def test_settings_import_fails_closed_when_production_root_password_is_missing():
    result = _run_settings_import(
        {
            "ENVIRONMENT": "production",
            "SECRET_KEY": "V8mQ2rL7xN4pK9dT6wC3zH5sJ1fB0uY8eA6iO4nP2rS9vX7cD5qG6hM3",
            "DEBUG": "False",
            "ALLOWED_HOSTS": "chatbot.example.com",
            "CORS_ALLOW_ALL_ORIGINS": "False",
            "CHATBOT_ALLOW_ANONYMOUS_LOCAL": "False",
            "MYSQL_PASSWORD": VALID_DATABASE_PASSWORD,
        },
        unset=("MYSQL_ROOT_PASSWORD",),
    )

    assert result.returncode != 0
    assert "MYSQL_ROOT_PASSWORD" in result.stderr


def test_settings_import_fails_closed_for_production_placeholders():
    base = {
        "ENVIRONMENT": "production",
        "DEBUG": "False",
        "ALLOWED_HOSTS": "chatbot.example.com",
        "CORS_ALLOW_ALL_ORIGINS": "False",
        "CHATBOT_ALLOW_ANONYMOUS_LOCAL": "False",
        "MYSQL_PASSWORD": VALID_DATABASE_PASSWORD,
        "MYSQL_ROOT_PASSWORD": VALID_DATABASE_PASSWORD,
    }
    for secret_key in ("replace-with-a-random-secret", "chatbot-secret-key-change-in-production"):
        result = _run_settings_import({**base, "SECRET_KEY": secret_key})
        assert result.returncode != 0
        assert "SECRET_KEY" in result.stderr


def test_settings_import_fails_closed_for_production_compose_password():
    result = _run_settings_import(
        {
            "ENVIRONMENT": "production",
            "SECRET_KEY": "V8mQ2rL7xN4pK9dT6wC3zH5sJ1fB0uY8eA6iO4nP2rS9vX7cD5qG6hM3",
            "DEBUG": "False",
            "ALLOWED_HOSTS": "chatbot.example.com",
            "CORS_ALLOW_ALL_ORIGINS": "False",
            "CHATBOT_ALLOW_ANONYMOUS_LOCAL": "False",
            "MYSQL_PASSWORD": "1234",
            "MYSQL_ROOT_PASSWORD": VALID_DATABASE_PASSWORD,
        }
    )

    assert result.returncode != 0
    assert "MYSQL_PASSWORD" in result.stderr


def test_settings_import_succeeds_with_explicit_secure_production_values():
    result = _run_settings_import(
        {
            "ENVIRONMENT": "production",
            "SECRET_KEY": "V8mQ2rL7xN4pK9dT6wC3zH5sJ1fB0uY8eA6iO4nP2rS9vX7cD5qG6hM3",
            "DEBUG": "False",
            "ALLOWED_HOSTS": "chatbot.example.com",
            "CORS_ALLOW_ALL_ORIGINS": "False",
            "CHATBOT_ALLOW_ANONYMOUS_LOCAL": "False",
            "MYSQL_PASSWORD": VALID_DATABASE_PASSWORD,
            "MYSQL_ROOT_PASSWORD": VALID_DATABASE_PASSWORD,
        }
    )

    assert result.returncode == 0, result.stderr
