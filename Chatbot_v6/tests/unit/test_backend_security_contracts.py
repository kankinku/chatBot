import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = REPO_ROOT / "Server" / "backend" / "chatbot_backend" / "settings.py"
ENV_EXAMPLE_PATH = REPO_ROOT / "Server" / "backend" / "env.example"
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
CONFIGURATION_SECURITY_PATH = (
    REPO_ROOT / "Server" / "backend" / "chatbot_backend" / "configuration_security.py"
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
    assert "ALLOWED_HOSTS=localhost,127.0.0.1" in source
    assert "CORS_ALLOW_ALL_ORIGINS=False" in source
    assert "CHATBOT_ALLOW_ANONYMOUS_LOCAL=True" in source
    assert "MYSQL_PASSWORD=1234" not in source


def test_docker_compose_declares_its_nonproduction_environment():
    source = COMPOSE_PATH.read_text(encoding="utf-8")
    assert "- ENVIRONMENT=development" in source
