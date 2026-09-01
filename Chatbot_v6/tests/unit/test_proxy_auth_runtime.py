"""Runtime authorization checks against Django's API client and ORM."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "Server" / "backend"

RUNTIME_SCRIPT = r'''
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "chatbot_backend.settings")

import django

django.setup()

from django.core.management import call_command
from django.db import connections
from django.test import Client, RequestFactory
from types import SimpleNamespace


connections.databases["default"] = {
    "ENGINE": "django.db.backends.sqlite3",
    "NAME": ":memory:",
    "ATOMIC_REQUESTS": False,
    "AUTOCOMMIT": True,
    "CONN_MAX_AGE": 0,
    "CONN_HEALTH_CHECKS": False,
    "OPTIONS": {},
    "TIME_ZONE": None,
    "USER": "",
    "PASSWORD": "",
    "HOST": "",
    "PORT": "",
    "TEST": {
        "CHARSET": None,
        "COLLATION": None,
        "NAME": None,
        "MIRROR": None,
        "MIGRATE": True,
        "SERIALIZE": True,
    },
}
try:
    del connections._connections.default
except AttributeError:
    pass

call_command("migrate", verbosity=0, interactive=False)

from chatbot_proxy import views
from chatbot_proxy.models import ChatLog, ChatMetrics, Conversation
from chatbot_proxy.security import PermissionDenied
from django.contrib.auth import get_user_model
from django.db.migrations.recorder import MigrationRecorder


client = Client()
assert client.get("/api/chatbot/status").status_code == 401
assert client.get("/api/chatbot/logs").status_code == 401

views.sync_make_chatbot_request = lambda **kwargs: {"status": "ok"}
assert client.get("/api/chatbot/health").status_code == 200

User = get_user_model()
regular_user = User.objects.create_user(username="regular")
operator_user = User.objects.create_user(username="operator", is_staff=True)
client.force_login(regular_user)
assert client.get("/api/chatbot/status").status_code == 403
client.force_login(operator_user)
assert client.get("/api/chatbot/status").status_code == 200

request = RequestFactory().get("/api/chatbot/status")
request.user = SimpleNamespace(
    is_authenticated=True,
    pk=7,
    is_staff=False,
    is_superuser=False,
)
request.META["REMOTE_ADDR"] = "127.0.0.1"
try:
    views._require_actor(request, operator=True)
except Exception as exc:
    assert getattr(exc, "status_code", None) == 403
else:
    raise AssertionError("a non-operator must not pass the operator boundary")

Conversation.objects.create(session_id="owned", owner_key="user:1")
Conversation.objects.create(session_id="other", owner_key="user:2")
Conversation.objects.create(session_id="legacy", owner_key=None)
try:
    views.get_or_create_conversation("owned", "user:2")
except PermissionDenied:
    pass
else:
    raise AssertionError("owner mismatch must be denied")
try:
    views.get_or_create_conversation("legacy", "user:2")
except PermissionDenied:
    pass
else:
    raise AssertionError("legacy rows without an owner must be denied")

list_request = RequestFactory().get("/api/chatbot/conversations")
list_request.user = SimpleNamespace(
    is_authenticated=True,
    pk=1,
    is_staff=False,
    is_superuser=False,
)
list_request.META["REMOTE_ADDR"] = "127.0.0.1"
conversations = views.get_conversations(list_request)
assert [item.session_id for item in conversations] == ["owned"]

assert Conversation._meta.get_field("owner_key").null is True
assert ChatLog._meta.get_field("owner_key").null is True
assert ChatMetrics._meta.get_field("owner_key").null is True
assert MigrationRecorder(connections["default"]).migration_qs.filter(
    app="chatbot_proxy", name="0002_add_owner_key"
).exists()

original_save_chat_log = views.save_chat_log
views.save_chat_log = lambda *args, **kwargs: (_ for _ in ()).throw(
    RuntimeError("expected telemetry failure")
)
views._safe_save_chat_log(level="INFO", message="x", owner_key="user:1")
views.save_chat_log = original_save_chat_log

original_update_chat_metrics = views.update_chat_metrics
views.update_chat_metrics = lambda *args, **kwargs: (_ for _ in ()).throw(
    RuntimeError("expected telemetry failure")
)
views._safe_update_chat_metrics("owned", True, 0.1, owner_key="user:1")
views.update_chat_metrics = original_update_chat_metrics

client.force_login(regular_user)
views.sync_make_chatbot_request = lambda **kwargs: {
    "answer": "runtime answer",
    "confidence": 0.9,
    "sources": [],
    "metrics": {},
    "fallback_used": False,
}
views.save_chat_log = lambda *args, **kwargs: (_ for _ in ()).throw(
    RuntimeError("expected request telemetry failure")
)
views.update_chat_metrics = lambda *args, **kwargs: (_ for _ in ()).throw(
    RuntimeError("expected request telemetry failure")
)
response = client.post(
    "/api/chatbot/ask",
    data={"question": "runtime question", "mode": "accuracy", "k": "auto"},
    content_type="application/json",
    HTTP_X_SESSION_ID="runtime-session",
)
assert response.status_code == 200, response.content
assert response.json()["answer"] == "runtime answer"
'''


def test_proxy_auth_policy_works_through_django_client_and_orm():
    environment = {
        name: os.environ[name]
        for name in ("PATH", "SystemRoot", "WINDIR", "TEMP", "TMP")
        if name in os.environ
    }
    environment.update(
        {
            "DJANGO_SETTINGS_MODULE": "chatbot_backend.settings",
            "ENVIRONMENT": "development",
            "SECRET_KEY": "V8mQ2rL7xN4pK9dT6wC3zH5sJ1fB0uY8eA6iO4nP2rS9vX7cD5qG6hM3",
            "DEBUG": "True",
            "ALLOWED_HOSTS": "testserver,localhost,127.0.0.1",
            "CORS_ALLOW_ALL_ORIGINS": "False",
            "CHATBOT_ALLOW_ANONYMOUS_LOCAL": "False",
            "MYSQL_PASSWORD": "Q7mR2xV9kL4pN8dT6wC3zH5sJ1fB0uY8eA6iO4nP2rS9vX7cD5qG6hM3",
        }
    )

    with tempfile.TemporaryDirectory() as isolated_cwd:
        isolated_root = Path(isolated_cwd)
        shutil.copytree(
            BACKEND_ROOT / "chatbot_backend",
            isolated_root / "chatbot_backend",
        )
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(isolated_root), str(BACKEND_ROOT))
        )
        result = subprocess.run(
            [sys.executable, "-c", RUNTIME_SCRIPT],
            cwd=isolated_cwd,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    assert result.returncode == 0, result.stderr
