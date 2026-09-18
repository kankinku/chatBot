"""Django proxy HTTP response contract tests.

The proxy depends on optional Django/Ninja runtime packages, so these tests
validate the public schema wiring without importing the application.
"""

import ast
from pathlib import Path


VIEWS_PATH = Path(__file__).resolve().parents[2] / "Server" / "backend" / "chatbot_proxy" / "views.py"


def _load_views_tree() -> ast.Module:
    return ast.parse(VIEWS_PATH.read_text(encoding="utf-8"), filename=str(VIEWS_PATH))


def _route_response_name(function: ast.FunctionDef) -> str | None:
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "response" and isinstance(keyword.value, ast.Name):
                return keyword.value.id
    return None


def test_proxy_response_models_have_distinct_names():
    tree = _load_views_tree()
    class_names = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]

    assert class_names.count("ChatMessageResponse") == 0
    assert class_names.count("SimpleChatResponse") == 1
    assert class_names.count("StoredChatMessageResponse") == 1


def test_chat_routes_reference_the_matching_response_models():
    tree = _load_views_tree()
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    assert _route_response_name(functions["proxy_simple_chat"]) == "SimpleChatResponse"
    assert _route_response_name(functions["get_conversation_detail"]) == "ConversationDetailResponse"

    simple_return_models = {
        node.func.id
        for node in ast.walk(functions["proxy_simple_chat"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    stored_message_models = {
        node.func.id
        for node in ast.walk(functions["get_conversation_detail"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "SimpleChatResponse" in simple_return_models
    assert "StoredChatMessageResponse" in stored_message_models

    detail_response = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ConversationDetailResponse"
    )
    messages_annotation = next(
        node.annotation for node in detail_response.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "messages"
    )
    assert isinstance(messages_annotation, ast.Subscript)
    assert isinstance(messages_annotation.value, ast.Name)
    assert messages_annotation.value.id == "List"
    assert isinstance(messages_annotation.slice, ast.Name)
    assert messages_annotation.slice.id == "StoredChatMessageResponse"
