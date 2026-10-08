from __future__ import annotations

import pytest
from greennode_agentbase import RequestContext
from greennode_agentbase.exceptions import GreenNodeRequestError
from langchain_core.messages import AIMessage, ToolMessage

from app import service
from app.config import get_settings


@pytest.fixture(autouse=True)
def _hitl_env(monkeypatch):
    monkeypatch.setenv("HITL_TOOLS", '["remember"]')
    get_settings.cache_clear()


def ctx(session="hitl-1"):
    return RequestContext(session_id=session, user_id="u1")


def _call(fact="Likes tea"):
    return AIMessage(
        "I will remember that.",
        tool_calls=[{"name": "remember", "args": {"fact": fact}, "id": "c1"}],
    )


async def test_interrupt_then_approve(fake_llm):
    fake_llm(_call(), AIMessage("Remembered."))
    out = await service.handle({"message": "Remember that I like tea"}, ctx())
    assert out["status"] == "interrupted"
    assert out["interrupt"]["tool_calls"][0]["name"] == "remember"

    # Awaiting approval => new chat is blocked
    with pytest.raises(GreenNodeRequestError) as e:
        await service.handle({"message": "hello"}, ctx())
    assert e.value.status_code == 409

    out = await service.handle(
        {"type": "resume", "decisions": [{"tool_call_id": "c1", "action": "approve"}]}, ctx()
    )
    assert out["status"] == "success"
    assert out["tools_used"] == ["remember"]


async def test_edit_args(fake_llm):
    model = fake_llm(_call("wrong"), AIMessage("ok"))
    await service.handle({"message": "remember"}, ctx("hitl-2"))
    await service.handle(
        {
            "type": "resume",
            "decisions": [{"tool_call_id": "c1", "action": "edit", "args": {"fact": "correct"}}],
        },
        ctx("hitl-2"),
    )
    tool_msgs = [m for m in model.calls[-1] if isinstance(m, ToolMessage)]
    assert "correct" in tool_msgs[-1].content


async def test_reject_returns_error_tool_message(fake_llm):
    model = fake_llm(_call(), AIMessage("Understood, I won't save it."))
    await service.handle({"message": "remember"}, ctx("hitl-3"))
    out = await service.handle(
        {
            "type": "resume",
            "decisions": [{"tool_call_id": "c1", "action": "reject", "reason": "not wanted"}],
        },
        ctx("hitl-3"),
    )
    assert out["status"] == "success"
    tool_msgs = [m for m in model.calls[-1] if isinstance(m, ToolMessage)]
    assert tool_msgs[-1].status == "error" and "not wanted" in tool_msgs[-1].content


async def test_resume_without_pending_is_409(fake_llm):
    fake_llm(AIMessage("ok"))
    with pytest.raises(GreenNodeRequestError) as e:
        await service.handle(
            {"type": "resume", "decisions": [{"tool_call_id": "x", "action": "approve"}]},
            ctx("hitl-4"),
        )
    assert e.value.status_code == 409


def test_placeholder_and_waiting_detection():
    from types import SimpleNamespace

    from app.hitl import is_placeholder_tool_message, is_waiting_approval

    ph = ToolMessage(
        "Tool call 'remember' with id 'c1' was interrupted before completion.",
        tool_call_id="c1",
        status="error",
    )
    assert is_placeholder_tool_message(ph)
    assert not is_placeholder_tool_message(ToolMessage("Remembered", tool_call_id="c1"))
    # AgentBaseMemoryEvents may return interrupts=() but next=('approval',)
    assert is_waiting_approval(SimpleNamespace(interrupts=(), next=("approval",)))
    assert not is_waiting_approval(SimpleNamespace(interrupts=(), next=()))


async def test_invalid_edit_is_rejected_not_executed(fake_llm):
    """`remember` takes {"fact": str}; an edit with the wrong shape must not reach the tool."""
    model = fake_llm(_call("x"), AIMessage("ok"))
    await service.handle({"message": "remember"}, ctx("hitl-5"))
    await service.handle(
        {
            "type": "resume",
            "decisions": [{"tool_call_id": "c1", "action": "edit", "args": {"wrong": 1}}],
        },
        ctx("hitl-5"),
    )
    tool_msgs = [m for m in model.calls[-1] if isinstance(m, ToolMessage)]
    assert (
        tool_msgs[-1].status == "error" and "Edited arguments are invalid" in tool_msgs[-1].content
    )


def test_edited_args_checked_against_mcp_json_schema():
    """MCP tools carry a JSON-Schema dict the adapter does not enforce — validate it ourselves."""
    from langchain_core.tools import StructuredTool

    from app.hitl import edited_args_error

    async def _noop(**kwargs):
        return "ok"

    schema = {
        "type": "object",
        "properties": {"amount": {"type": "integer"}, "to": {"type": "string"}},
        "required": ["amount", "to"],
        "additionalProperties": False,
    }
    tool = StructuredTool(name="pay", description="pay", args_schema=schema, coroutine=_noop)
    assert edited_args_error(tool, {"amount": 10, "to": "bob"}) is None
    assert edited_args_error(tool, {"amount": "lots", "to": "attacker"})
    assert edited_args_error(tool, {})
    assert edited_args_error(tool, {"amount": 1, "to": "b", "extra": True})


async def test_unmatched_hitl_pattern_is_reported(monkeypatch, caplog):
    import app.tools as tools_mod
    from app.auth.inbound import Principal
    from app.tools import collect_tools

    tools_mod._reported_unmatched.clear()  # warned once per pattern per process
    monkeypatch.setenv("HITL_TOOLS", '["remember", "github_create-issue"]')
    get_settings.cache_clear()
    await collect_tools(get_settings(), Principal(user_id="u1"))
    assert "github_create-issue" in caplog.text and "approval NOT enforced" in caplog.text
