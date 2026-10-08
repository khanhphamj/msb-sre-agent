"""Standard LangGraph for an AgentBase agent.

    START -> compress -> recall -> [route] -> agent ─┬─(no tool call)─────> [reflect] ──> END
                                          │                         └─(fail)─> agent
                                          ├─(needs approval)──> approval ──> tools ──> agent
                                          └─(regular tool)────────────────> tools ──> agent

- compress : compress context (rolling summary) when over the token budget. (agentbase-build-memory)
- recall   : auto-recall long-term memory for the latest question.         (agentbase-build-memory)
- route    : (LLM_ADAPTIVE_ROUTING) small model scores complexity ⇒ picks the agent tier. (agentbase-build-llm)
- agent    : call the LLM per lane (agent | agent_simple | agent_complex → tier). (agentbase-build-llm)
- approval : HITL interrupt() for tools matching HITL_TOOLS.               (agentbase-build-hitl)
- tools    : local / memory / MCP tools; rejected tool => error ToolMessage. (agentbase-build-mcp)
- reflect  : self-evaluation, only when REFLECTION_ENABLED.                (agentbase-build-eval)

Add business nodes (router, planner, sub-agent...) by extending build_graph;
do NOT write a new graph elsewhere.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config import Settings
from app.graph.state import AgentState
from app.hitl import build_approval_node, build_tools_node, requires_approval
from app.llm import get_llm
from app.llm.routing import TASK_BY_COMPLEXITY, classify
from app.memory.compression import fit_to_budget, summarize_if_needed
from app.memory.long_term import auto_recall, get_ltm
from app.prompts import get_system_prompt
from app.reflection import build_reflect_node, route_after_reflect


def _last_human_text(messages: list[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.text
    return ""


def build_system_message(state: AgentState, system_prompt: str) -> SystemMessage:
    # Stable order (static first, dynamic last) to maximize provider prompt-cache hits.
    parts = [system_prompt]
    if summary := state.get("summary"):
        parts.append(f"## Summary of the earlier conversation\n{summary}")
    if recalled := state.get("recalled"):
        parts.append("## Known facts about the user\n" + "\n".join(f"- {f}" for f in recalled))
    if critique := state.get("critique"):
        parts.append(f"## Feedback on the previous answer (answer again, better)\n{critique}")
    return SystemMessage("\n\n".join(parts))


def build_graph(
    *, settings: Settings, tools: list[BaseTool], checkpointer: BaseCheckpointSaver
) -> CompiledStateGraph:
    # Model per LANE: "agent" (default large); adaptive routing picks agent_simple/agent/agent_complex
    llms: dict[str, Any] = {}

    def llm_for(task: str):
        if task not in llms:
            llms[task] = get_llm(task, tools)
        return llms[task]

    ltm = get_ltm(settings) if settings.ltm_auto_recall else None
    system_prompt, langfuse_prompt = get_system_prompt()
    # Link the prompt version ONLY to the agent's generation: metadata passed at call time ⇒ attached to the chain
    # `llm.<task>` (direct parent of the generation). Putting it in node/graph metadata ⇒ judge/summarize
    # generations get the wrong prompt linked too (seen on real Langfuse).
    prompt_meta: dict[str, Any] = {"langfuse_prompt": langfuse_prompt} if langfuse_prompt else {}

    async def compress(state: AgentState) -> dict:
        update = await summarize_if_needed(state["messages"], state.get("summary", ""), settings)
        return {**(update or {}), "critique": "", "reflection_round": 0, "llm_task": "agent"}

    async def recall(state: AgentState, config: RunnableConfig) -> dict:
        if ltm is None:
            return {"recalled": []}
        user_id = config["configurable"]["actor_id"]
        facts = await auto_recall(
            ltm, user_id, _last_human_text(state["messages"]), settings.ltm_recall_limit
        )
        return {"recalled": facts}

    async def route(state: AgentState) -> dict:
        level = await classify(_last_human_text(state["messages"]), state.get("summary", ""))
        return {"llm_task": TASK_BY_COMPLEXITY[level]}

    async def agent(state: AgentState) -> dict:
        messages = fit_to_budget(
            [build_system_message(state, system_prompt), *state["messages"]], settings
        )
        task = state.get("llm_task") or "agent"
        response = await llm_for(task).ainvoke(messages, config={"metadata": prompt_meta})
        return {"messages": [response]}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            if any(requires_approval(tc["name"], settings) for tc in last.tool_calls):
                return "approval"
            return "tools"
        if not settings.reflection_enabled:
            return END
        if settings.reflection_skip_simple and state.get("llm_task") == "agent_simple":
            return END  # simple message: not worth 1–2 extra LLM calls (measured on Langfuse: +27s)
        return "reflect"

    builder = StateGraph(AgentState)
    builder.add_node("compress", compress)
    builder.add_node("recall", recall)
    builder.add_node("agent", agent)
    builder.add_node("approval", build_approval_node(settings, tools))
    builder.add_node("tools", build_tools_node(tools))
    builder.add_node("reflect", build_reflect_node(settings))

    builder.add_edge(START, "compress")
    builder.add_edge("compress", "recall")
    if (
        settings.llm_adaptive_routing
    ):  # route runs only at turn start; tool rounds returning to agent keep the tier
        builder.add_node("route", route)
        builder.add_edge("recall", "route")
        builder.add_edge("route", "agent")
    else:
        builder.add_edge("recall", "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["approval", "tools", "reflect", END])
    builder.add_edge("approval", "tools")
    builder.add_edge("tools", "agent")
    builder.add_conditional_edges("reflect", route_after_reflect, ["agent", END])
    return builder.compile(checkpointer=checkpointer)


def recursion_limit(settings: Settings) -> int:
    # 4 turn-start nodes (compress, recall, route?, agent) + 3 steps per tool round + reflection rounds
    return 4 + 3 * settings.max_tool_rounds + 2 * (settings.reflection_max_retries + 1) + 1
