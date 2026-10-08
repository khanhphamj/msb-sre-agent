"""Outbound credentials from AgentBase Identity (Access Control) for tools.

Which helper (see the agentbase-build-identity skill):
  - Agent-wide secret, same for every user ⇒ `agent_api_key` (Static API Key provider) /
    `agent_access_token` (OAuth2 provider, client-credentials M2M).
  - The END USER's own credential ⇒ `user_access_token` (OAuth2 3LO consent) / `user_api_key` (Delegated
    API Key provider). They wrap the SDK so a missing consent does NOT block the request: the SDK's default
    poller waits up to 600s for the user (> REQUEST_TIMEOUT_S) — instead we raise `AuthorizationRequired`, the
    tool returns the link, the user consents, and the NEXT call gets the credential immediately.

Rules:
  - Decorate an INNER function, never the @tool: the injected key/token must not appear in the tool schema the
    LLM sees, nor in tool-call traces.
  - Per-user credentials are keyed by GreenNodeAgentBaseContext user id — app.service sets it from the verified
    Principal on every path. No user ⇒ refuse (never fetch a credential for "nobody").
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any, TypeVar

from greennode_agentbase import GreenNodeAgentBaseContext, requires_access_token, requires_api_key
from greennode_agentbase.core.config import get_config_value
from greennode_agentbase.identity import TokenPoller
from greennode_agentbase.identity import auth as _sdk_auth

from app.config import get_settings
from app.observability import tracing

log = logging.getLogger(__name__)
# The SDK logs the end-user id ("Agent user ID: …") at INFO on every 3LO call — keep it out of app logs
logging.getLogger("greennode_agentbase.identity.auth").setLevel(logging.WARNING)
T = TypeVar("T")
Decorator = Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]

AUTH_REQUIRED_PREFIX = "AUTHORIZATION_REQUIRED"
_auth_url: ContextVar[str | None] = ContextVar("identity_auth_url", default=None)
_shared_client: Any = None


class AuthorizationRequired(Exception):
    """The user must grant access first. `tool_message` (also `str(e)`) is what the tool returns to the LLM —
    so even a tool without try/except (ToolNode handle_tool_errors) still shows the link."""

    def __init__(self, provider_name: str, url: str | None):
        self.provider_name = provider_name
        self.url = url
        super().__init__(self.tool_message)

    @property
    def tool_message(self) -> str:
        return (
            f"{AUTH_REQUIRED_PREFIX}: the user must grant access to '{self.provider_name}' first. "
            f"Give the user this link to open: {self.url} — after they finish, ask them to send the request "
            "again. Do not retry before that."
        )


class _ReturnLinkPoller(TokenPoller):
    """Replaces the SDK's 600s polling: stop right away and hand the link back to the user."""

    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    async def poll_for_token(self) -> str:
        raise AuthorizationRequired(self.provider_name, _auth_url.get())


async def _remember_auth_url(url: str) -> None:
    # Async on purpose: the SDK awaits async callbacks in the request's context (a sync one would run in a
    # copied context and the url would be lost).
    _auth_url.set(url)


def _client() -> Any:
    """One IdentityClient per process: its IAM token cache is per instance, and requires_api_key would
    otherwise build a new client (= a new IAM token request) on every call."""
    global _shared_client
    if _shared_client is None:
        _shared_client = _sdk_auth.IdentityClient(iam_credentials=_sdk_auth.IAMCredentials())
    return _shared_client


def _require_agent_identity() -> None:
    # Without it the SDK silently CREATES a new workload identity on the platform (and writes it to
    # .greennode.json) — the providers you configured would not be attached to it.
    name = get_config_value("GREENNODE_AGENT_IDENTITY")
    if not name or name.startswith("<"):
        raise ValueError(
            'GREENNODE_AGENT_IDENTITY is not set (Runtime injects it; locally put "agent_identity" in '
            ".greennode.json) — the name of the identity your providers belong to (/agentbase-identity)."
        )


def _require_user_and_callback() -> str:
    if not GreenNodeAgentBaseContext.get_user_id():
        raise PermissionError(
            "Per-user credentials need an authenticated user (no user id in context)."
        )
    url = get_settings().identity_callback_url
    if not url:
        raise ValueError(
            "IDENTITY_CALLBACK_URL is required for per-user credentials (and must be in the agent "
            "identity's allowedReturnUrls)."
        )
    return url


def _wrap(
    kind: str, provider_name: str, make_sdk: Callable[[], Callable], per_user: bool
) -> Decorator:
    """`make_sdk()` builds the SDK decorator at CALL time (per-user kwargs depend on the request)."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            _require_agent_identity()
            if per_user:
                _auth_url.set(None)
            decorated = make_sdk()(func)
            needs_auth: AuthorizationRequired | None = None
            with tracing.step(f"identity.{kind}", input={"provider": provider_name}) as st:
                try:
                    result = await decorated(*args, **kwargs)
                    st.set(output={"status": "ok"})  # never the credential itself
                except AuthorizationRequired as e:
                    # Expected, not an error: record WARNING and raise AFTER the span closes (an exception
                    # escaping tracing.step would mark the span ERROR)
                    st.set(output={"status": "authorization_required"}, level="WARNING")
                    needs_auth = e
            if needs_auth is not None:
                raise needs_auth
            return result

        return wrapper

    return decorator


def agent_api_key(provider_name: str, *, into: str = "api_key") -> Decorator:
    """The agent's own key from a Static API Key provider (same for every user)."""
    return _wrap(
        "agent_api_key",
        provider_name,
        lambda: requires_api_key(
            provider_name=provider_name, auth_flow="M2M", into=into, identity_client=_client()
        ),
        per_user=False,
    )


def agent_access_token(
    provider_name: str, scopes: list[str], *, into: str = "access_token"
) -> Decorator:
    """The agent's own OAuth2 token (client credentials / M2M — same for every user)."""
    return _wrap(
        "agent_access_token",
        provider_name,
        lambda: requires_access_token(
            provider_name=provider_name, scopes=scopes, auth_flow="M2M", into=into
        ),
        per_user=False,
    )


def user_access_token(
    provider_name: str, scopes: list[str], *, into: str = "access_token"
) -> Decorator:
    """The end user's OAuth2 token (3LO / USER_FEDERATION). Raises AuthorizationRequired until they consent."""

    def make() -> Callable:
        return requires_access_token(
            provider_name=provider_name,
            scopes=scopes,
            auth_flow="USER_FEDERATION",
            into=into,
            callback_url=_require_user_and_callback(),
            on_auth_url=_remember_auth_url,
            token_poller=_ReturnLinkPoller(provider_name),
        )

    return _wrap("user_access_token", provider_name, make, per_user=True)


def user_api_key(provider_name: str, *, into: str = "api_key") -> Decorator:
    """The end user's own API key (Delegated API Key provider). Raises AuthorizationRequired until provided."""

    def make() -> Callable:
        return requires_api_key(
            provider_name=provider_name,
            auth_flow="USER_FEDERATION",
            into=into,
            identity_client=_client(),
            callback_url=_require_user_and_callback(),
            on_auth_url=_remember_auth_url,
            token_poller=_ReturnLinkPoller(provider_name),
        )

    return _wrap("user_api_key", provider_name, make, per_user=True)
