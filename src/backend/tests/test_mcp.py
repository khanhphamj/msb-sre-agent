from __future__ import annotations

import base64
import json

from langchain_core.tools import StructuredTool, ToolException

from app.tools.mcp import _guard_policy, iam_principal


def _tool(behavior):
    async def coro(query: str):
        return behavior(query)

    return StructuredTool.from_function(
        coroutine=coro,
        name="tavily_search",
        description="search",
        response_format="content_and_artifact",
        handle_tool_error=True,  # like langchain-mcp-adapters tools,
    )


async def test_policy_denied_exception_becomes_clear_message():
    def deny(_):
        raise ToolException("Request denied by policy.")

    tool = _guard_policy(_tool(deny), "tavily")
    out = await tool.ainvoke({"query": "x"})
    assert out.startswith("POLICY_DENIED") and "Do NOT call" in out


async def test_normal_result_passthrough_and_other_errors():
    ok = _guard_policy(_tool(lambda q: (f"result {q}", None)), "tavily")
    assert await ok.ainvoke({"query": "a"}) == "result a"

    def boom(_):
        raise ToolException("upstream timeout")

    other = _guard_policy(_tool(boom), "tavily")
    assert "upstream timeout" in await other.ainvoke({"query": "a"})  # other errors: unchanged


def test_iam_principal_from_token_sub():
    payload = base64.urlsafe_b64encode(json.dumps({"sub": "abc-123"}).encode()).decode().rstrip("=")
    assert iam_principal(f"h.{payload}.s") == "iam:abc-123"
    assert iam_principal("garbage") is None


async def test_denied_connector_short_circuits_other_tools():
    from app.tools.mcp import reset_policy_denials

    calls = []

    def deny(q):
        calls.append(q)
        raise ToolException("Request denied by policy.")

    reset_policy_denials()
    t1, t2 = _guard_policy(_tool(deny), "stock"), _guard_policy(_tool(deny), "stock")
    assert (await t1.ainvoke({"query": "a"})).startswith("POLICY_DENIED")
    assert (await t2.ainvoke({"query": "b"})).startswith("POLICY_DENIED")
    assert calls == ["a"]  # 2nd tool no longer calls the gateway
    reset_policy_denials()  # new request ⇒ retry normally
    await t2.ainvoke({"query": "c"})
    assert calls == ["a", "c"]


async def test_failing_server_negative_cached(monkeypatch, tmp_path):
    import json

    import app.tools.mcp as mcp_mod
    from app.auth.inbound import Principal
    from app.config import get_settings

    calls = []

    class _Boom:
        def __init__(self, *a, **k):
            pass

        async def get_tools(self, server_name):
            calls.append(server_name)
            raise ConnectionError("upstream down")

    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"servers": {"down": {"url": "https://x/mcp", "auth": "none"}}}))
    monkeypatch.setenv("MCP_CONFIG_FILE", str(cfg))
    get_settings.cache_clear()
    monkeypatch.setattr(mcp_mod, "MultiServerMCPClient", _Boom)
    mcp_mod._failed.clear()
    errors: dict = {}
    assert await mcp_mod.get_mcp_tools(get_settings(), Principal(user_id="u"), errors) == []
    assert await mcp_mod.get_mcp_tools(get_settings(), Principal(user_id="u"), errors) == []
    assert calls == ["down"]  # 2nd time skipped thanks to negative cache
    assert "skipped" in errors["down"]


async def test_per_user_server_failure_not_cached_for_others(monkeypatch, tmp_path):
    import json

    import app.tools.mcp as mcp_mod
    from app.auth.inbound import Principal
    from app.config import get_settings

    class _Boom:
        def __init__(self, *a, **k):
            pass

        async def get_tools(self, server_name):
            raise ConnectionError("401 Unauthorized")

    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"servers": {"gh": {"url": "https://x/mcp", "auth": "user_jwt"}}}))
    monkeypatch.setenv("MCP_CONFIG_FILE", str(cfg))
    get_settings.cache_clear()
    monkeypatch.setattr(mcp_mod, "MultiServerMCPClient", _Boom)
    mcp_mod._failed.clear()
    await mcp_mod.get_mcp_tools(get_settings(), Principal(user_id="a", token="t-a"), {})
    assert "gh" not in mcp_mod._failed  # user A's error does not block user B


async def test_policy_text_in_successful_result_is_not_a_deny():
    out = await _guard_policy(
        _tool(lambda q: ("Article: what is Request denied by policy?", None)), "web"
    ).ainvoke({"query": "x"})
    assert out.startswith("Article")


# --- mcp_servers.json `auth` fails closed: the agent's IAM token is never sent by accident
def _load(monkeypatch, tmp_path, servers: dict) -> dict:
    import json

    import app.tools.mcp as mcp_mod
    from app.config import get_settings

    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"servers": servers}))
    monkeypatch.setenv("MCP_CONFIG_FILE", str(cfg))
    get_settings.cache_clear()
    return mcp_mod.load_mcp_config(get_settings())


def test_server_without_auth_is_skipped_not_given_iam(monkeypatch, tmp_path, caplog):
    loaded = _load(monkeypatch, tmp_path, {"thirdparty": {"url": "https://mcp.example.com/mcp"}})
    assert loaded == {}
    assert '"auth" is required' in caplog.text


def test_authorization_header_conflicts_with_iam(monkeypatch, tmp_path, caplog):
    entry = {"url": "https://x/mcp", "auth": "iam", "headers": {"Authorization": "Bearer k"}}
    assert _load(monkeypatch, tmp_path, {"notes": entry}) == {}
    assert "conflicts with auth" in caplog.text


def test_static_key_with_auth_none_is_sent_and_no_iam(monkeypatch, tmp_path):
    import app.tools.mcp as mcp_mod
    from app.auth.inbound import Principal
    from app.config import get_settings

    monkeypatch.setenv("NOTES_MCP_KEY", "k-123")
    entry = {"url": "http://localhost:8080/mcp", "auth": "none"}
    entry["headers"] = {"Authorization": "Bearer ${NOTES_MCP_KEY}"}
    cfg = _load(monkeypatch, tmp_path, {"notes": entry})["notes"]
    conn = mcp_mod._connection(cfg, Principal(user_id="u"), get_settings())
    assert conn["headers"]["Authorization"] == "Bearer k-123"
    assert "auth" not in conn  # no IAMBearerAuth attached


def test_stdio_needs_no_auth_and_iam_to_non_gateway_warns(monkeypatch, tmp_path, caplog):
    servers = {
        "fetch": {"transport": "stdio", "command": "uvx", "args": ["mcp-server-fetch"]},
        "odd": {"url": "https://mcp.example.com/mcp", "auth": "iam"},
        "gw": {"url": "https://gw-a-b.agentbase-gateway.aiplatform.vngcloud.vn/x", "auth": "iam"},
    }
    assert set(_load(monkeypatch, tmp_path, servers)) == {"fetch", "odd", "gw"}
    assert "'odd' uses auth=iam" in caplog.text and "'gw' uses auth=iam" not in caplog.text
