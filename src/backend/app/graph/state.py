from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    # Rolling summary of the compressed part of the conversation (persisted in the checkpoint)
    summary: str
    # Long-term memories auto-recalled for the current turn (overwritten every turn)
    recalled: list[str]
    # HITL: tool_call_id -> rejection reason (cleared after the tools node runs)
    hitl_rejected: dict[str, str]
    # Self-evaluation loop: judge feedback for the retry answer, number of retries so far
    critique: str
    reflection_round: int
    # LLM lane for the current turn (adaptive routing): agent_simple | agent | agent_complex
    llm_task: str
