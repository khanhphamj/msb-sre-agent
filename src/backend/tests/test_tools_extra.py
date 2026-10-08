from __future__ import annotations

from greennode_agentbase import RequestContext
from langchain_core.messages import AIMessage

from app import service
from app.config import get_settings


def test_search_knowledge_cites_source(tmp_path, monkeypatch):
    import app.tools.local_tools as lt

    (tmp_path / "annual-leave.md").write_text(
        "# Annual leave\n\nFull-time employees get 12 days of annual leave.\n\n# Benefits\n\nHealth insurance.",
        encoding="utf-8",
    )
    monkeypatch.setenv("KNOWLEDGE_DIR", str(tmp_path))
    get_settings.cache_clear()
    lt._knowledge_chunks.cache_clear()
    assert "search_knowledge" in [t.name for t in lt.get_local_tools()]
    out = lt.search_knowledge.invoke({"query": "how many days of annual leave"})
    assert "12 days of annual leave" in out and "[Source: annual-leave.md" in out
    lt._knowledge_chunks.cache_clear()


def test_no_knowledge_dir_means_no_tool(monkeypatch, tmp_path):
    import app.tools.local_tools as lt

    monkeypatch.setenv("KNOWLEDGE_DIR", str(tmp_path / "missing"))
    get_settings.cache_clear()
    lt._knowledge_chunks.cache_clear()
    assert "search_knowledge" not in [t.name for t in lt.get_local_tools()]
    lt._knowledge_chunks.cache_clear()


async def test_forwarded_claims_reach_tools_only_allowlisted(fake_llm, monkeypatch):
    import app.memory.short_term as st

    seen = {}
    orig = st.thread_config

    def spy(settings, **kw):
        cfg = orig(settings, **kw)
        seen.update(cfg["configurable"])
        return cfg

    monkeypatch.setattr("app.service.thread_config", spy)
    monkeypatch.setenv("AUTH_FORWARD_CLAIMS", '["employee_id"]')
    get_settings.cache_clear()
    fake_llm(AIMessage("ok"))
    import app.service as sv

    run = sv._Run(
        sv.Principal(user_id="alice", claims={"employee_id": "E01", "secret": "x"}),
        "claims-1",
        "chat",
        {"messages": []},
        "x",
    )
    await run._prepare()
    assert seen["user_claims"] == {"employee_id": "E01"}


def test_pii_masked_in_traces():
    from app.observability import tracing

    out = tracing._mask(data="Contact a.b@vng.com.vn or 0912 345 678, CCCD 012345678901")
    assert "vng.com.vn" not in out and "345 678" not in out and "012345678901" not in out


async def test_max_tool_rounds_gives_friendly_error(fake_llm, monkeypatch):
    import pytest
    from greennode_agentbase.exceptions import GreenNodeRequestError

    monkeypatch.setenv("MAX_TOOL_ROUNDS", "1")
    get_settings.cache_clear()
    loop = AIMessage("", tool_calls=[{"name": "get_current_time", "args": {}, "id": "t"}])
    fake_llm(*[loop] * 20)
    with pytest.raises(GreenNodeRequestError) as e:
        await service.handle({"message": "loop"}, RequestContext(session_id="loop-1", user_id="u"))
    assert e.value.status_code == 422


def test_forbidden_tools_evaluator():
    from evals.evaluators import forbidden_tools

    meta = {"forbidden_tools": ["create_leave_request"]}
    assert forbidden_tools(output={"tools_used": []}, metadata=meta).value == 1.0
    assert (
        forbidden_tools(output={"tools_used": ["create_leave_request"]}, metadata=meta).value == 0.0
    )
    interrupted = {
        "status": "interrupted",
        "interrupt": {"tool_calls": [{"name": "create_leave_request"}]},
    }
    assert forbidden_tools(output=interrupted, metadata=meta).value == 0.0
