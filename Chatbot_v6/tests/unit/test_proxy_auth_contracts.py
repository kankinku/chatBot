"""Static contracts for the R2-B2 route authorization boundary."""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VIEWS_PATH = REPO_ROOT / "Server" / "backend" / "chatbot_proxy" / "views.py"
MODELS_PATH = REPO_ROOT / "Server" / "backend" / "chatbot_proxy" / "models.py"
MIGRATION_PATH = (
    REPO_ROOT
    / "Server"
    / "backend"
    / "chatbot_proxy"
    / "migrations"
    / "0002_add_owner_key.py"
)
URLS_PATH = REPO_ROOT / "Server" / "backend" / "chatbot_backend" / "urls.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in _tree(path).body
        if isinstance(node, ast.FunctionDef)
    }


def _calls(function: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def _has_actor_call(function: ast.FunctionDef, *, operator: bool = False) -> bool:
    for call in _calls(function, "_require_actor"):
        operator_keyword = next(
            (keyword for keyword in call.keywords if keyword.arg == "operator"),
            None,
        )
        if not operator:
            return operator_keyword is None
        if (
            operator_keyword is not None
            and isinstance(operator_keyword.value, ast.Constant)
            and operator_keyword.value.value is True
        ):
            return True
    return False


def test_sensitive_proxy_routes_require_an_actor():
    functions = _functions(VIEWS_PATH)
    for name in (
        "proxy_simple_chat",
        "proxy_ai_question",
        "proxy_batch_questions",
        "get_conversations",
        "get_conversation_detail",
        "delete_conversation",
    ):
        assert _has_actor_call(functions[name]), name


def test_operational_proxy_routes_require_operator():
    functions = _functions(VIEWS_PATH)
    for name in ("proxy_process_pdfs", "proxy_metrics", "get_chat_logs", "get_chat_metrics"):
        assert _has_actor_call(functions[name], operator=True), name


def test_detailed_status_requires_operator_but_health_remains_public():
    functions = _functions(VIEWS_PATH)
    assert _has_actor_call(functions["proxy_chatbot_status"], operator=True)
    assert not _calls(functions["proxy_health_check"], "_require_actor")


def test_route_boundary_translates_policy_failures_to_http_errors():
    functions = _functions(VIEWS_PATH)
    helper = functions["_require_actor"]

    assert _calls(helper, "resolve_actor")
    policy_exceptions = {
        node.type.id
        for node in ast.walk(helper)
        if isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Name)
    }
    assert {"AuthenticationRequired", "PermissionDenied"}.issubset(policy_exceptions)
    http_statuses = {
        call.args[0].value
        for call in _calls(helper, "HttpError")
        if len(call.args) >= 1
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, int)
    }
    assert {401, 403}.issubset(http_statuses)


def test_conversation_access_is_owner_filtered_and_legacy_rows_are_not_claimed():
    functions = _functions(VIEWS_PATH)
    for name in ("get_conversations", "get_conversation_detail", "delete_conversation"):
        filter_calls = [
            call
            for call in ast.walk(functions[name])
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "filter"
        ]
        assert any(
            any(keyword.arg == "owner_key" for keyword in call.keywords)
            for call in filter_calls
        ), name

    conversation_helper = functions["get_or_create_conversation"]
    assert any(
        isinstance(node, ast.arg) and node.arg == "owner_key"
        for node in ast.walk(conversation_helper)
    )
    assert _calls(conversation_helper, "require_owner")
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Actor"
        for node in ast.walk(conversation_helper)
    )


def test_ai_question_translates_owner_policy_failure_to_http_403():
    function = _functions(VIEWS_PATH)["proxy_ai_question"]
    assert any(
        isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Name)
        and node.type.id == "PermissionDenied"
        for node in ast.walk(function)
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "HttpError"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == 403
        for node in ast.walk(function)
    )


def test_persistence_helpers_require_owner_key():
    functions = _functions(VIEWS_PATH)
    for name in ("save_chat_log", "update_chat_metrics"):
        function = functions[name]
        assert any(
            isinstance(node, ast.arg) and node.arg == "owner_key"
            for node in ast.walk(function)
        ), name
        assert any(
            isinstance(node, ast.keyword) and node.arg == "owner_key"
            for node in ast.walk(function)
        ), name


def test_telemetry_writes_have_best_effort_wrappers():
    functions = _functions(VIEWS_PATH)
    for wrapper_name, helper_name in (
        ("_safe_save_chat_log", "save_chat_log"),
        ("_safe_update_chat_metrics", "update_chat_metrics"),
    ):
        wrapper = functions[wrapper_name]
        assert _calls(wrapper, helper_name), wrapper_name
        assert any(
            isinstance(node, ast.ExceptHandler)
            and isinstance(node.type, ast.Name)
            and node.type.id == "Exception"
            for node in ast.walk(wrapper)
        ), wrapper_name

    proxy = functions["proxy_ai_question"]
    assert not _calls(proxy, "save_chat_log")
    assert not _calls(proxy, "update_chat_metrics")
    assert len(_calls(proxy, "_safe_save_chat_log")) >= 3
    assert len(_calls(proxy, "_safe_update_chat_metrics")) >= 3

    delete_route = functions["delete_conversation"]
    assert not _calls(delete_route, "save_chat_log")
    assert _calls(delete_route, "_safe_save_chat_log")


def test_owner_key_is_additive_on_all_sensitive_models():
    tree = _tree(MODELS_PATH)
    classes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    for name in ("Conversation", "ChatLog", "ChatMetrics"):
        assert any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "owner_key"
                for target in node.targets
            )
            for node in classes[name].body
        ), name


def test_owner_key_migration_is_nullable_and_does_not_claim_existing_rows():
    source = MIGRATION_PATH.read_text(encoding="utf-8")
    assert "migrations.AddField" in source
    assert source.count("name='owner_key'") == 3
    assert source.count("null=True") >= 3
    assert source.count("blank=True") >= 3
    assert "RunPython" not in source


def test_root_pdf_routes_require_operator():
    source = URLS_PATH.read_text(encoding="utf-8")
    assert source.count("_require_actor(request, operator=True)") >= 2
