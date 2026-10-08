"""Adaptive routing (optional, LLM_ADAPTIVE_ROUTING): pick the agent's tier by question complexity.

    simple   → task agent_simple  (small)      : greetings, single-step, 1-tool lookup, definitions
    standard → task agent         (large)      : normal conversation, a few tool calls
    complex  → task agent_complex (reasoning)  : multi-step, comparison/analysis, calculation, planning

The router uses the small tier (1 short call, ~100 tokens). Error/unparseable ⇒ "standard" (safe).
Enable only for mixed traffic: mostly simple questions (savings) + some needing reasoning (quality).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm, model_for, tier_for
from app.observability import tracing

log = logging.getLogger(__name__)
Complexity = Literal["simple", "standard", "complex"]
TASK_BY_COMPLEXITY: dict[str, str] = {
    "simple": "agent_simple",
    "standard": "agent",
    "complex": "agent_complex",
}

ROUTER_PROMPT = """Classify the complexity of the request to pick a suitable model. Return only JSON
{"complexity": "simple"|"standard"|"complex", "reason": "<short>"}.
- simple: greetings, thanks, single-step/single-fact questions, simple lookups.
- standard: normal conversation, needs a few steps or a few tools.
- complex: multi-step reasoning, analysis/comparison across sources, calculation, planning, writing/reviewing code,
  questions with complex constraints."""


async def classify(text: str, summary: str = "") -> Complexity:
    context = f"Conversation context (summary): {summary}\n\n" if summary else ""
    with tracing.step("llm.route", input={"text": text[:500]}) as st:
        try:
            out = await get_llm("router").ainvoke(
                [SystemMessage(ROUTER_PROMPT), HumanMessage(f"{context}Request: {text}")],
                config={"run_name": "llm.router"},
            )
            match = re.search(r"\{.*\}", str(out.content), re.S)
            data = json.loads(match.group(0)) if match else {}
            level = data.get("complexity")
            level = level if level in TASK_BY_COMPLEXITY else "standard"
        except Exception as e:  # noqa: BLE001 — a router error must not break the chat turn
            st.set(level="WARNING", status_message=f"{type(e).__name__}: {e}")
            return "standard"
        st.set(
            output={
                "complexity": level,
                "task": TASK_BY_COMPLEXITY[level],
                "reason": data.get("reason"),
            }
        )
        task = TASK_BY_COMPLEXITY[level]
        log.info("llm.route complexity=%s task=%s model=%s", level, task, model_for(tier_for(task)))
        return level  # type: ignore[return-value]
