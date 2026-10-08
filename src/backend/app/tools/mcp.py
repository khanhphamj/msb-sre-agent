"""Connect external tools via MCP.

Connection standard (declared in mcp_servers.json, supports ${ENV_VAR}):
  - Prod: agent -> AgentBase MCP Gateway (MCP Connector) -> MCP server. The gateway handles inbound auth,
    outbound credentials (Identity providerName) and Policy Group. Create with /agentbase-gateway.
  - Local: may connect directly to an MCP server (streamable_http or stdio) via "envs": ["local"].

Auth type per server (`auth`) — REQUIRED for HTTP servers, there is no default (a default of "iam" would send the
agent's platform credential to any URL that forgot it):
  - "iam"      : the agent's IAM Bearer token (gateway inboundAuth.mode = IAM). Token auto-refreshes.
                 Only for the AgentBase MCP Gateway (*.agentbase-gateway.aiplatform.vngcloud.vn).
  - "user_jwt" : forwards the end-user's JWT (gateway inboundAuth.mode = JWT) => per-request.
  - "none"     : no auth from the agent; a static key for a self-built MCP server goes in
                 "headers": {"Authorization": "Bearer ${MY_KEY}"} (only allowed with "none").
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from collections.abc import AsyncGenerator, Generator
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.auth.inbound import Principal
from app.config import Settings
from app.observability import tracing

log = logging.getLogger(__name__)

IAM_TOKEN_URL = "https://iam.api.vngcloud.vn/accounts-api/v2/auth/token"


def iam_principal(token: str) -> str | None:
    """Principal that Policy Group uses for an IAM caller = `iam:<sub>` of the IAM token (the service
    account's user id, NOT the client_id). Only decodes the payload for logging; no signature check."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        sub = json.loads(base64.urlsafe_b64decode(payload)).get("sub")
        return f"iam:{sub}" if sub else None
    except (IndexError, ValueError):
        return None


class IAMBearerAuth(httpx.Auth):
    """httpx.Auth that fetches an IAM token (client_credentials) from GREENNODE_CLIENT_ID/SECRET, cached until exp.

    On AgentBase Runtime these variables are injected automatically; locally read from env/.greennode.json.
    """

    def __init__(self) -> None:
        from greennode_agentbase import IAMCredentials

        self._creds = IAMCredentials()
        self._token: str | None = None
        self._exp = 0.0
        self.principal: str | None = None

    def _request_token(self) -> httpx.Request:
        self._creds.require()
        return httpx.Request(
            "POST",
            IAM_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "client_credentials"},
        )

    def _store(self, resp: httpx.Response) -> None:
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._exp = time.time() + int(data.get("expires_in", 1800)) * 0.9
        if self.principal is None:
            self.principal = iam_principal(self._token)
            log.info("IAM principal of the agent (use in Policy Group): %s", self.principal)

    def _valid(self) -> bool:
        return bool(self._token) and time.time() < self._exp

    def _basic(self) -> httpx.BasicAuth:
        return httpx.BasicAuth(self._creds.client_id or "", self._creds.client_secret or "")

    def refresh(self) -> None:
        """Fetch a new token now (sync). Raises httpx.HTTPStatusError if IAM rejects the credentials."""
        with httpx.Client(auth=self._basic(), timeout=30) as c:
            self._store(c.send(self._request_token()))

    def sync_auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response]:
        if not self._valid():
            self.refresh()
        request.headers["Authorization"] = f"Bearer {self._token}"
        yield request

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        if not self._valid():
            async with httpx.AsyncClient(auth=self._basic(), timeout=30) as c:
                self._store(await c.send(self._request_token()))
        request.headers["Authorization"] = f"Bearer {self._token}"
        yield request


_iam_auth: IAMBearerAuth | None = None


def _get_iam_auth() -> IAMBearerAuth:
    global _iam_auth
    if _iam_auth is None:
        _iam_auth = IAMBearerAuth()
    return _iam_auth


_AUTH_MODES = ("iam", "user_jwt", "none")
_GATEWAY_HOST_SUFFIX = ".agentbase-gateway.aiplatform.vngcloud.vn"


def mcp_config_error(cfg: dict[str, Any]) -> str | None:
    """Why this server entry must not be loaded (None = OK). Fails closed: never guess an auth mode."""
    if cfg.get("transport") == "stdio":
        return None  # local process, no HTTP auth
    auth = cfg.get("auth")
    if auth not in _AUTH_MODES:
        return f'"auth" is required, one of {list(_AUTH_MODES)} (got {auth!r})'
    if auth != "none" and any(k.lower() == "authorization" for k in cfg.get("headers") or {}):
        return f'headers.Authorization conflicts with auth="{auth}" (it would be overwritten); use auth="none"'
    return None


def load_mcp_config(settings: Settings) -> dict[str, dict[str, Any]]:
    path = Path(settings.mcp_config_file)
    if not path.exists():
        return {}
    raw = json.loads(os.path.expandvars(path.read_text(encoding="utf-8")))
    servers = {}
    for name, cfg in (raw.get("servers") or {}).items():
        envs = cfg.get("envs")
        if envs and settings.app_env not in envs:
            continue
        if cfg.get("enabled", True) is False:
            continue
        if err := mcp_config_error(cfg):
            log.error("MCP server %r skipped — invalid mcp_servers.json entry: %s", name, err)
            continue
        host = urlparse(cfg.get("url", "")).hostname or ""
        if cfg.get("auth") == "iam" and not host.endswith(_GATEWAY_HOST_SUFFIX):
            log.warning(
                "MCP server %r uses auth=iam but %r is not an AgentBase MCP Gateway — the agent's IAM token "
                "is sent there. Use auth=none (+ headers) for other servers.",
                name,
                host,
            )
        servers[name] = cfg
    return servers


def _connection(cfg: dict[str, Any], principal: Principal, settings: Settings) -> dict[str, Any]:
    transport = cfg.get("transport", "streamable_http")
    if transport == "stdio":
        return {
            "transport": "stdio",
            "command": cfg["command"],
            "args": cfg.get("args", []),
            "env": cfg.get("env"),
        }
    conn: dict[str, Any] = {
        "transport": transport,
        "url": cfg["url"],
        "headers": dict(cfg.get("headers") or {}),
        "timeout": settings.mcp_tool_timeout_s,
        # The adapter default is very long ⇒ a "silent" MCP stream holds the request for minutes
        "sse_read_timeout": settings.mcp_tool_timeout_s,
    }
    auth = cfg.get("auth", "none")
    if auth == "iam":
        conn["auth"] = _get_iam_auth()
    elif auth == "user_jwt":
        if not principal.token:
            raise PermissionError("MCP server requires a user JWT but the request has no token")
        conn["headers"]["Authorization"] = f"Bearer {principal.token}"
    return conn


# Connector already blocked by policy in the current request ⇒ short-circuit its other tools
# (seen in practice: a reasoning model tried 8 blocked tools in turn ⇒ 84s). Set is recreated per request.
_DENIED_SERVERS: ContextVar[set[str] | None] = ContextVar("mcp_denied_servers", default=None)


def reset_policy_denials() -> None:
    """Call at the start of each request (service) — the set is shared by all LangGraph nodes/tasks of the request."""
    _DENIED_SERVERS.set(set())


POLICY_DENIED_MARKERS = ("denied by policy", "no policy allows this request")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


def _guard_policy(tool: BaseTool, server: str) -> BaseTool:
    """The gateway returns a policy deny as MCP error "Request denied by policy." — the adapter raises
    ToolException, LangChain turns it into plain text => the LLM thinks it is transient and retries many
    times. Turn it into a clear message + trace event `mcp.policy_denied` (WARNING) on Langfuse.
    """
    if getattr(tool, "_policy_guarded", False) or tool.coroutine is None:
        return tool
    original = tool.coroutine

    def _denied(short_circuit: bool = False) -> Any:
        denied = _DENIED_SERVERS.get()
        if denied is not None:
            denied.add(server)
        tracing.event(
            "mcp.policy_denied",
            level="WARNING",
            metadata={"server": server, "tool": tool.name, "short_circuit": short_circuit},
        )
        msg = (
            f"POLICY_DENIED: the agent is not allowed to use tools of connector '{server}' (blocked by "
            "the MCP Gateway Policy Group). Do NOT call any tool of this connector again this turn; "
            "answer with the information available and tell the user permission is missing."
        )
        return (msg, None) if tool.response_format == "content_and_artifact" else msg

    def _is_denied(text: str) -> bool:
        return any(m in text.lower() for m in POLICY_DENIED_MARKERS)

    async def guarded(*args: Any, **kwargs: Any) -> Any:
        if server in (_DENIED_SERVERS.get() or ()):
            return _denied(short_circuit=True)  # no redundant request to the gateway
        try:
            result = await original(*args, **kwargs)
        except Exception as e:  # adapter raises ToolException when MCP returns isError
            if _is_denied(str(e)):
                return _denied()
            raise
        # Do NOT scan successful result content: a web page/document containing "denied by policy" would
        # lock the whole connector (prompt-injection attack). The gateway returns denies via the error path (ToolException).
        return result

    tool.coroutine = guarded
    object.__setattr__(tool, "_policy_guarded", True)
    return tool


# Cache the tool list for user-independent servers (iam/none) to avoid tools/list on every request.
_static_tools: dict[str, tuple[float, list[BaseTool]]] = {}
_STATIC_TTL_S = 300


_failed: dict[
    str, tuple[float, str]
] = {}  # negative cache: failed server ⇒ skipped for MCP_FAILURE_TTL_S


def _is_auth_error(e: BaseException) -> bool:
    if isinstance(e, PermissionError):
        return True
    for x in getattr(e, "exceptions", None) or [e]:
        resp = getattr(x, "response", None)
        if getattr(resp, "status_code", None) in (401, 403):
            return True
        if any(code in str(x) for code in ("401", "403", "Unauthorized", "Forbidden")):
            return True
    return False


async def _load_server(
    name: str, cfg: dict, settings: Settings, principal: Principal, errors: dict[str, str] | None
) -> list[BaseTool]:
    auth = cfg.get("auth", "none")
    per_user = auth == "user_jwt"
    cached = _static_tools.get(name)
    hit = not per_user and cached is not None and time.time() - cached[0] < _STATIC_TTL_S
    failed = None if per_user else _failed.get(name)
    skip = failed is not None and time.time() - failed[0] < settings.mcp_failure_ttl_s
    span_input = {
        "server": name,
        "transport": cfg.get("transport", "streamable_http"),
        "url": cfg.get("url"),
        "auth": auth,
        "cached": hit,
        "skipped_recent_failure": skip,
    }
    with tracing.step("mcp.list_tools", input=span_input) as st:
        if hit:
            server_tools = cached[1]
        elif skip:
            # Seen in practice: a failing connector made every request wait ~15s ⇒ no retry within the TTL
            if errors is not None:
                errors[name] = f"skipped (failed {int(time.time() - failed[0])}s ago): {failed[1]}"
            st.set(level="WARNING", status_message=f"skipped: {failed[1]}")
            return []
        else:
            try:
                client = MultiServerMCPClient(
                    {name: _connection(cfg, principal, settings)}, tool_name_prefix=True
                )
                server_tools = await asyncio.wait_for(
                    client.get_tools(server_name=name), timeout=settings.mcp_list_timeout_s
                )
            except Exception as e:  # noqa: BLE001 — one failing MCP server must not crash the agent
                msg = f"{type(e).__name__}: {str(e)[:300]}"
                log.warning("MCP server '%s' unavailable, skipping: %s", name, msg)
                # Negative cache ONLY for shared servers + infra errors. A per-user server (user_jwt) or
                # one user's permission/401 error must not make the server disappear for every user.
                if not per_user and not _is_auth_error(e):
                    _failed[name] = (time.time(), msg)
                if errors is not None:
                    errors[name] = msg
                st.set(level="WARNING", status_message=msg)
                return []
            _failed.pop(name, None)
            server_tools = [_guard_policy(t, name) for t in server_tools]
            if allow := cfg.get("allow_tools"):
                prefix = f"{name}_"
                server_tools = [t for t in server_tools if t.name.removeprefix(prefix) in allow]
            if not per_user:
                _static_tools[name] = (time.time(), server_tools)
        st.set(output=[t.name for t in server_tools])
        return server_tools


async def get_mcp_tools(
    settings: Settings, principal: Principal, errors: dict[str, str] | None = None
) -> list[BaseTool]:
    """Load tools from ALL servers IN PARALLEL (one slow server does not add up latency)."""
    servers = load_mcp_config(settings)
    results = await asyncio.gather(
        *(_load_server(n, c, settings, principal, errors) for n, c in servers.items())
    )
    return [t for group in results for t in group]
