"""Self-evaluation loop (evaluator–optimizer) — optional, OFF by default (REFLECTION_ENABLED).

    agent (final answer) -> reflect --pass--> END
                              └--fail & retries left--> agent (with critique) -> reflect ...

- The judge uses the `judge` task (default tier: large) and returns JSON {"pass": bool, "score": 0..1, "critique": "..."}.
- A rejected answer is REMOVED from history (RemoveMessage) so it doesn't pollute short-term memory.
- Each judgment writes a `self_eval` score to the Langfuse trace => see the online quality distribution.
- Costs 1 extra LLM call per turn (+2 per retry: agent + judge): enable only for use cases that need high accuracy.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph import END

from app.config import Settings
from app.llm import get_llm
from app.observability import tracing

JUDGE_PROMPT = """You are a judge of the quality of an AI agent's answers.
Criteria: {criteria}
Return JSON only: {{"pass": true|false, "score": <0..1>, "critique": "<short feedback for fixing it>"}}"""
DEFAULT_CRITIQUE = "The answer did not meet the criteria. Make it accurate, complete and on point."


def _as_bool(v, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "pass", "đạt")
    if isinstance(v, int | float):
        return bool(v)
    return default


def _as_score(v, default: float) -> float:
    try:
        return min(max(float(v), 0.0), 1.0)
    except (TypeError, ValueError):
        return default


def _parse(text: str) -> dict:
    """Judge returns malformed JSON (null, "false" as a string, missing fields) ⇒ don't crash, default to PASS."""
    match = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return {
        "pass": _as_bool(data.get("pass"), True),
        "score": _as_score(data.get("score"), 1.0),
        "critique": str(data.get("critique") or ""),
    }


def build_reflect_node(settings: Settings):
    async def reflect(state: dict) -> dict:
        messages = state["messages"]
        answer = messages[-1]
        question = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        judge = get_llm("judge")  # in-turn self-eval flow (default tier: large)
        try:
            result = await judge.ainvoke(
                [
                    SystemMessage(JUDGE_PROMPT.format(criteria=settings.reflection_criteria)),
                    HumanMessage(
                        f"Question:\n{question.text if question else ''}\n\nAnswer:\n{answer.text}"
                    ),
                ],
                config={"run_name": "reflection.judge", "tags": ["evaluation"]},
            )
        except Exception as e:  # noqa: BLE001 — a judge failure must not break the existing answer
            tracing.event(
                "reflection.judge_failed",
                level="WARNING",
                metadata={"error": f"{type(e).__name__}: {str(e)[:200]}"},
            )
            return {"critique": "", "reflection_round": 0}
        verdict = _parse(str(result.content))
        rounds = state.get("reflection_round", 0)
        tracing.score_trace("self_eval", verdict["score"], comment=verdict["critique"])
        if verdict["pass"] or rounds >= settings.reflection_max_retries:
            return {"critique": "", "reflection_round": 0}
        # The answer is removed below, so a retry MUST follow: routing keys on a non-empty critique, and a
        # judge that says "fail" without one would otherwise end the turn with an empty reply.
        critique = verdict["critique"] or DEFAULT_CRITIQUE
        return {
            "messages": [RemoveMessage(id=answer.id)],
            "critique": critique,
            "reflection_round": rounds + 1,
        }

    return reflect


def route_after_reflect(state: dict) -> str:
    return "agent" if state.get("critique") else END
