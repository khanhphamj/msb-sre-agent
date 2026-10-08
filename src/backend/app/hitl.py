"""Human-in-the-loop (HITL) — a human approves tool calls before they execute.

Mechanism: LangGraph `interrupt()` + checkpointer. The graph stops at the `approval` node, state is stored
in AgentBase Memory (AgentBaseMemoryEvents); the client sends decisions via the payload
`{"type": "resume", "decisions": [...]}` in the same session => the graph continues exactly where it stopped.

    agent --(tool call matches HITL_TOOLS)--> approval --interrupt--> [client approves] --> tools
    agent --(other tool call)----------------------------------------------------------> tools

Interrupt payload sent to the client:
    {"type": "tool_approval", "tool_calls": [{"id", "name", "args"}], "message": "<AI text>"}
Decision sent back by the client (one decision per tool_call_id; missing => reject):
    {"tool_call_id": "...", "action": "approve" | "edit" | "reject", "args": {...}, "reason": "..."}

Rules:
- HITL_TOOLS is a list of globs on tool names (e.g. ["gateway_*_delete", "send_email"]).
  Apply it by default to side-effecting tools (write/delete/send/payment). Read-only tools need no approval.
- The approval node is re-run FROM THE START on resume => put no side effects before `interrupt()`.
- While awaiting approval, new chats in the same session are rejected (409) — resume first.
"""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from app.config import Settings
from app.observability import tracing

VALID_ACTIONS = {"approve", "edit", "reject"}


def requires_approval(tool_name: str, settings: Settings) -> bool:
    return any(fnmatch(tool_name, pattern) for pattern in settings.hitl_tools)


def last_ai(messages: list) -> AIMessage | None:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m
    return None


def validate_decisions(decisions: Any) -> list[dict]:
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("decisions must be a non-empty list")
    for d in decisions:
        if not isinstance(d, dict) or not d.get("tool_call_id"):
            raise ValueError("each decision needs a tool_call_id")
        if d.get("action") not in VALID_ACTIONS:
            raise ValueError(f"action must be one of {sorted(VALID_ACTIONS)}")
        if d["action"] == "edit" and not isinstance(d.get("args"), dict):
            raise ValueError("action=edit requires args (object)")
    return decisions


def edited_args_error(tool: Any, args: dict) -> str | None:
    """Why `args` don't fit the tool's input schema (None = valid). Uses `tool_call_schema` — the schema the LLM
    sees, without injected args. MCP tools carry a JSON-Schema dict that the adapter does NOT enforce, so an edit
    like {"amount": "lots"} would otherwise reach the MCP server unchecked."""
    schema = getattr(tool, "tool_call_schema", None) if tool is not None else None
    if schema is None:
        return None
    try:
        if isinstance(schema, dict):
            import jsonschema

            jsonschema.validate(args, schema)
        else:
            schema.model_validate(args)
    except Exception as e:  # noqa: BLE001 — jsonschema.ValidationError / pydantic.ValidationError
        return (str(getattr(e, "message", "")) or str(e)).splitlines()[0][:200]
    return None


def build_approval_node(settings: Settings, tools: list | None = None):
    by_name = {t.name: t for t in tools or []}

    async def approval(state: dict) -> dict:
        ai = last_ai(state["messages"])
        pending = [
            {"id": tc["id"], "name": tc["name"], "args": tc["args"]}
            for tc in ai.tool_calls
            if requires_approval(tc["name"], settings)
        ]
        tracing.event("hitl.request", input={"tool_calls": pending})
        decisions = interrupt({"type": "tool_approval", "tool_calls": pending, "message": ai.text})
        by_id = {d["tool_call_id"]: d for d in validate_decisions(decisions)}
        tracing.event("hitl.decision", output={"decisions": list(by_id.values())})

        new_calls, rejected = [], {}
        for tc in ai.tool_calls:
            if not requires_approval(tc["name"], settings):
                new_calls.append(tc)
                continue
            d = by_id.get(tc["id"], {"action": "reject", "reason": "no decision"})
            if d["action"] == "edit":
                if err := edited_args_error(by_name.get(tc["name"]), d["args"]):
                    rejected[tc["id"]] = f"Edited arguments are invalid ({err})"
                    tracing.event(
                        "hitl.invalid_edit", level="WARNING", metadata={"tool": tc["name"]}
                    )
                else:
                    tc = {**tc, "args": d["args"]}
            elif d["action"] == "reject":
                rejected[tc["id"]] = d.get("reason") or "Rejected by the user"
            new_calls.append(tc)
        # Same id => add_messages REPLACES the old AIMessage (keeps the edited args)
        updated = AIMessage(content=ai.content, tool_calls=new_calls, id=ai.id)
        return {"messages": [updated], "hitl_rejected": rejected}

    return approval


def build_tools_node(tools: list):
    """ToolNode wrapped with handling for rejected tool calls (returns an error ToolMessage to the LLM)."""
    tool_node = ToolNode(tools, handle_tool_errors=True)

    async def run_tools(state: dict, config: RunnableConfig) -> dict:
        ai = last_ai(state["messages"])
        rejected: dict[str, str] = state.get("hitl_rejected") or {}
        to_run = [tc for tc in ai.tool_calls if tc["id"] not in rejected]
        out: list = []
        if to_run:
            result = await tool_node.ainvoke(
                {**state, "messages": [AIMessage(content="", tool_calls=to_run)]}, config
            )
            out.extend(result["messages"])
        for tc in ai.tool_calls:
            if tc["id"] in rejected:
                out.append(
                    ToolMessage(
                        content=f"Not executed: {rejected[tc['id']]}. Ask the user again if needed.",
                        tool_call_id=tc["id"],
                        name=tc["name"],
                        status="error",
                    )
                )
        return {"messages": out, "hitl_rejected": {}}

    return run_tools


def interrupt_payload(interrupts: Any) -> dict | None:
    """Normalize an Interrupt (from ainvoke/astream `__interrupt__`, or a snapshot) for the client."""
    if not interrupts:
        return None
    first = interrupts[0]
    return {"id": getattr(first, "id", None), **(getattr(first, "value", None) or {})}


def is_waiting_approval(snapshot: Any) -> bool:
    """Whether the session is awaiting approval.

    Seen in practice with AgentBaseMemoryEvents: reading state RIGHT after the graph stops may not show the
    `__interrupt__` write yet (interrupts=()) even though `next=('approval',)`. So rely mainly on `next`.
    """
    return bool(getattr(snapshot, "interrupts", None)) or "approval" in (
        getattr(snapshot, "next", None) or ()
    )


def is_placeholder_tool_message(message: Any) -> bool:
    """AgentBaseMemoryEvents inserts a placeholder ToolMessage for an "orphaned" tool_call when loading a checkpoint
    without an interrupt ("Tool call 'x' with id 'y' was interrupted before completion.")."""
    content = getattr(message, "content", "")
    return (
        getattr(message, "status", None) == "error"
        and isinstance(content, str)
        and content.startswith("Tool call '")
        and content.endswith("was interrupted before completion.")
    )
