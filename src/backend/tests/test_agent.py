from __future__ import annotations

import pytest
from greennode_agentbase import RequestContext
from greennode_agentbase.exceptions import GreenNodeRequestError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app import service
from app.config import get_settings
from app.memory.compression import _safe_cut_index, summarize_if_needed


def ctx(session: str = "s1", user: str = "u1", headers: dict | None = None) -> RequestContext:
    return RequestContext(session_id=session, user_id=user, request_headers=headers)


async def test_chat_with_tool_call_and_memory(fake_llm):
    fake_llm(
        AIMessage(
            "", tool_calls=[{"name": "remember", "args": {"fact": "Likes coffee"}, "id": "c1"}]
        ),
        AIMessage("Noted that you like coffee."),
    )
    out = await service.handle({"message": "I like coffee"}, ctx())
    assert out["status"] == "success"
    assert out["response"] == "Noted that you like coffee."

    # Next turn: auto-recall puts the fact in the system prompt, history is kept per session
    model = fake_llm(AIMessage("You like coffee."))
    await service.handle({"message": "What do I like to drink? coffee or tea"}, ctx())
    system = model.calls[-1][0].content
    assert "Likes coffee" in system
    assert any(
        isinstance(m, HumanMessage) and "I like coffee" in m.content for m in model.calls[-1]
    )


async def test_sessions_and_users_are_isolated(fake_llm):
    fake_llm(AIMessage("ok"))
    await service.handle({"message": "secret of u1"}, ctx(session="s1", user="u1"))
    model = fake_llm(AIMessage("ok"))
    await service.handle({"message": "hello"}, ctx(session="s1", user="u2"))
    assert not any("secret of u1" in str(m.content) for m in model.calls[-1])


async def test_stream_emits_tokens_and_done(fake_llm):
    fake_llm(AIMessage("Hello there"))
    gen = await service.handle({"message": "hi", "stream": True}, ctx(session="s-stream"))
    events = [e async for e in gen]
    assert events[-1]["event"] == "done"
    assert events[-1]["response"] == "Hello there"


async def test_validation_errors(fake_llm):
    fake_llm(AIMessage("ok"))
    with pytest.raises(GreenNodeRequestError):
        await service.handle({"message": ""}, ctx())
    with pytest.raises(GreenNodeRequestError):
        await service.handle({"message": "hi"}, ctx(session=None))


def test_cut_index_never_orphans_tool_message():
    msgs = [
        HumanMessage("a", id="1"),
        AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": "x"}], id="2"),
        ToolMessage("r", tool_call_id="x", id="3"),
        AIMessage("b", id="4"),
    ]
    assert _safe_cut_index(msgs, keep_last=2) == 1  # tail starts at AIMessage(tool_calls)


async def test_summarize_when_over_budget(fake_llm, monkeypatch):
    monkeypatch.setenv("CONTEXT_MAX_TOKENS", "50")
    monkeypatch.setenv("CONTEXT_KEEP_LAST", "2")
    get_settings.cache_clear()
    fake_llm(AIMessage("- summary"))
    msgs = [HumanMessage("x " * 100, id=str(i)) for i in range(6)]
    update = await summarize_if_needed(msgs, "", get_settings())
    assert update["summary"] == "- summary"
    assert len(update["messages"]) == 4


def test_auth_none_forbidden_outside_local(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("MEMORY_BACKEND", "agentbase")
    monkeypatch.setenv("MEMORY_ID", "m")
    monkeypatch.setenv(
        "MEMORY_STRATEGY_ID", "strat-1"
    )  # valid memory config: isolate the auth check
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="AUTH_MODE=none"):
        get_settings()


def test_iam_pair_comes_from_greennode_json(tmp_path, monkeypatch):
    import json

    from app.config import normalize_iam_env

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".greennode.json").write_text(json.dumps({"client_id": "A", "client_secret": "S"}))
    monkeypatch.setenv("GREENNODE_CLIENT_ID", "STRAY")  # stray variable from the shell
    monkeypatch.delenv("GREENNODE_CLIENT_SECRET", raising=False)
    normalize_iam_env()
    import os

    assert (os.environ["GREENNODE_CLIENT_ID"], os.environ["GREENNODE_CLIENT_SECRET"]) == ("A", "S")


def test_lone_iam_var_dropped_without_file(tmp_path, monkeypatch):
    import os

    from app.config import normalize_iam_env

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GREENNODE_CLIENT_ID", "STRAY")
    monkeypatch.delenv("GREENNODE_CLIENT_SECRET", raising=False)
    normalize_iam_env()
    assert "GREENNODE_CLIENT_ID" not in os.environ


def test_hard_trim_never_drops_current_question(monkeypatch):
    from langchain_core.messages import SystemMessage

    from app.memory.compression import fit_to_budget

    monkeypatch.setenv("CONTEXT_HARD_LIMIT_TOKENS", "500")
    get_settings.cache_clear()
    msgs = [
        SystemMessage("sys"),
        HumanMessage("current question", id="h1"),
        AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": "x"}], id="a1"),
        ToolMessage("X" * 50_000, tool_call_id="x", id="t1"),  # huge tool output
    ]
    out = fit_to_budget(msgs, get_settings())
    assert any(isinstance(m, HumanMessage) and m.content == "current question" for m in out)
    tool = next(m for m in out if isinstance(m, ToolMessage))
    assert "truncated" in tool.content and len(tool.content) < 10_000


async def test_summarizer_error_does_not_break_turn(fake_llm, monkeypatch):
    from app.memory.compression import summarize_if_needed

    monkeypatch.setenv("CONTEXT_MAX_TOKENS", "50")
    monkeypatch.setenv("CONTEXT_KEEP_LAST", "2")
    get_settings.cache_clear()

    class _Boom:
        async def ainvoke(self, *a, **k):
            raise RuntimeError("summarizer down")

    monkeypatch.setattr("app.memory.compression.get_llm", lambda task="agent", tools=None: _Boom())
    msgs = [HumanMessage("x " * 100, id=str(i)) for i in range(6)]
    assert await summarize_if_needed(msgs, "", get_settings()) is None


def test_placeholder_dropped_when_real_tool_message_exists():
    from app.memory.compression import drop_placeholder_tool_messages

    ph = ToolMessage(
        "Tool call 'remember' with id 'c1' was interrupted before completion.",
        tool_call_id="c1",
        status="error",
    )
    real = ToolMessage("Remembered", tool_call_id="c1")
    assert drop_placeholder_tool_messages([ph, real]) == [real]
    assert drop_placeholder_tool_messages([ph]) == [
        ph
    ]  # no real one yet => keep it (avoids an orphan tool_call)


def _streaming_models():
    import httpx
    import openai
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk

    class _DiesMidStream(GenericFakeChatModel):
        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            for t in ["Xin ", "chào ", "anh"]:
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=t))
                if run_manager:
                    await run_manager.on_llm_new_token(t, chunk=chunk)
                yield chunk
            raise openai.APIConnectionError(request=httpx.Request("POST", "https://llm.test"))

    return _DiesMidStream(messages=iter([])), GenericFakeChatModel, openai.APIConnectionError


async def test_fallback_mid_stream_sends_reset_before_new_answer(monkeypatch):
    """Primary dies after streaming tokens ⇒ fallback answers ⇒ client must get `reset` first, never
    'Xin chào anhHello there' glued together."""
    primary, Fake, conn_error = _streaming_models()
    backup = Fake(messages=iter([AIMessage("Hello there")]))
    llm = primary.with_fallbacks([backup], exceptions_to_handle=(conn_error,))
    monkeypatch.setattr("app.graph.builder.get_llm", lambda task="agent", tools=None: llm)
    gen = await service.handle({"message": "hi", "stream": True}, ctx(session="s-fallback"))
    events = [e async for e in gen]
    kinds = [e["event"] for e in events]
    assert kinds.count("reset") == 1
    after = events[kinds.index("reset") + 1 :]
    assert "".join(e["data"] for e in after if e["event"] == "token") == "Hello there"
    assert events[-1]["event"] == "done" and events[-1]["response"] == "Hello there"


async def test_normal_stream_has_no_reset(monkeypatch):
    _, Fake, _ = _streaming_models()
    model = Fake(messages=iter([AIMessage("Hello there friend")]))
    monkeypatch.setattr("app.graph.builder.get_llm", lambda task="agent", tools=None: model)
    gen = await service.handle({"message": "hi", "stream": True}, ctx(session="s-plain"))
    events = [e async for e in gen]
    assert "reset" not in [e["event"] for e in events]
    assert "".join(e["data"] for e in events if e["event"] == "token") == "Hello there friend"
