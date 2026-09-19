"""Authentication and ownership policy primitives for the proxy API.

This module intentionally has no Django dependency so the policy can be tested
in the repository's offline unit-test environment.
"""

from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any, Optional


class AuthenticationRequired(Exception):
    """Raised when a request has no acceptable authenticated actor."""


class PermissionDenied(Exception):
    """Raised when an actor cannot access or mutate a resource."""


@dataclass(frozen=True)
class Actor:
    """The stable identity and privilege level used by proxy authorization."""

    owner_key: str
    authenticated: bool
    is_operator: bool


def is_loopback_address(remote_addr: Optional[str]) -> bool:
    """Return whether *remote_addr* is a valid loopback IP address."""

    if not remote_addr:
        return False
    try:
        return ip_address(remote_addr).is_loopback
    except ValueError:
        return False


def resolve_actor(
    user: Any,
    remote_addr: Optional[str],
    *,
    debug: bool,
    allow_anonymous_local: bool,
) -> Actor:
    """Resolve a request actor under an explicit authentication policy.

    Anonymous access is limited to loopback development requests and must be
    explicitly enabled. It never grants operator privileges.
    """

    authenticated = bool(getattr(user, "is_authenticated", False))
    if authenticated:
        user_id = getattr(user, "pk", None)
        if user_id is None:
            user_id = getattr(user, "username", None)
        if user_id is None or str(user_id).strip() == "":
            raise AuthenticationRequired("authenticated user has no stable identifier")
        return Actor(
            owner_key=f"user:{user_id}",
            authenticated=True,
            is_operator=bool(
                getattr(user, "is_staff", False)
                or getattr(user, "is_superuser", False)
            ),
        )

    if debug and allow_anonymous_local and is_loopback_address(remote_addr):
        return Actor(
            owner_key=f"local:{remote_addr}",
            authenticated=False,
            is_operator=False,
        )

    raise AuthenticationRequired("authentication is required")


def require_owner(actor: Actor, owner_key: Optional[str]) -> Actor:
    """Return *actor* only when it owns a non-null resource."""

    if not owner_key or actor.owner_key != owner_key:
        raise PermissionDenied("resource is not owned by the request actor")
    return actor
