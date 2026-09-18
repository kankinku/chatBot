from types import SimpleNamespace

import pytest

from Server.backend.chatbot_proxy.security import (
    AuthenticationRequired,
    PermissionDenied,
    require_owner,
    resolve_actor,
)


def test_anonymous_request_is_denied_by_default():
    with pytest.raises(AuthenticationRequired):
        resolve_actor(
            SimpleNamespace(is_authenticated=False),
            "127.0.0.1",
            debug=True,
            allow_anonymous_local=False,
        )


def test_anonymous_local_request_requires_explicit_development_flag():
    actor = resolve_actor(
        SimpleNamespace(is_authenticated=False),
        "127.0.0.1",
        debug=True,
        allow_anonymous_local=True,
    )

    assert actor.owner_key == "local:127.0.0.1"
    assert actor.authenticated is False
    assert actor.is_operator is False


def test_anonymous_ipv6_loopback_is_allowed_only_in_development():
    actor = resolve_actor(
        SimpleNamespace(is_authenticated=False),
        "::1",
        debug=True,
        allow_anonymous_local=True,
    )

    assert actor.owner_key == "local:::1"


def test_anonymous_local_request_is_denied_when_debug_is_disabled():
    with pytest.raises(AuthenticationRequired):
        resolve_actor(
            SimpleNamespace(is_authenticated=False),
            "127.0.0.1",
            debug=False,
            allow_anonymous_local=True,
        )


@pytest.mark.parametrize("remote_addr", [None, "not-an-ip", ""])
def test_invalid_or_missing_remote_address_is_not_loopback(remote_addr):
    with pytest.raises(AuthenticationRequired):
        resolve_actor(
            SimpleNamespace(is_authenticated=False),
            remote_addr,
            debug=True,
            allow_anonymous_local=True,
        )


@pytest.mark.parametrize("remote_addr", ["8.8.8.8", "2001:4860:4860::8888"])
def test_anonymous_public_request_is_never_allowed(remote_addr):
    with pytest.raises(AuthenticationRequired):
        resolve_actor(
            SimpleNamespace(is_authenticated=False),
            remote_addr,
            debug=True,
            allow_anonymous_local=True,
        )


def test_authenticated_user_gets_stable_owner_key():
    actor = resolve_actor(
        SimpleNamespace(is_authenticated=True, pk=42, is_staff=False, is_superuser=False),
        "8.8.8.8",
        debug=False,
        allow_anonymous_local=False,
    )

    assert actor.owner_key == "user:42"
    assert actor.authenticated is True
    assert actor.is_operator is False


@pytest.mark.parametrize("attribute", ["is_staff", "is_superuser"])
def test_staff_or_superuser_is_operator(attribute):
    user = SimpleNamespace(
        is_authenticated=True,
        pk=7,
        is_staff=False,
        is_superuser=False,
    )
    setattr(user, attribute, True)

    assert resolve_actor(
        user,
        "127.0.0.1",
        debug=False,
        allow_anonymous_local=False,
    ).is_operator is True


def test_authenticated_user_without_stable_identifier_is_denied():
    with pytest.raises(AuthenticationRequired):
        resolve_actor(
            SimpleNamespace(is_authenticated=True, pk=None, username=None),
            "127.0.0.1",
            debug=False,
            allow_anonymous_local=False,
        )


def test_authenticated_user_can_fall_back_to_username_identifier():
    actor = resolve_actor(
        SimpleNamespace(is_authenticated=True, pk=None, username="hanji"),
        "127.0.0.1",
        debug=False,
        allow_anonymous_local=False,
    )

    assert actor.owner_key == "user:hanji"


def test_require_owner_rejects_missing_or_different_owner():
    actor = SimpleNamespace(owner_key="user:42")

    with pytest.raises(PermissionDenied):
        require_owner(actor, None)
    with pytest.raises(PermissionDenied):
        require_owner(actor, "user:7")


def test_require_owner_returns_actor_for_matching_owner():
    actor = SimpleNamespace(owner_key="user:42")

    assert require_owner(actor, "user:42") is actor
