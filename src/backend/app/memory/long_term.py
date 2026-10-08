"""Long-term memory = per-user facts/preferences, semantic search, shared across all sessions.

Two mechanisms, used together:
1. Auto-recall (node `recall` in the graph): every turn, search by the latest question and put
   the results into the system prompt. Doesn't depend on the LLM remembering to call a tool.
2. Tools `remember` / `recall_memory`: let the LLM save/look up proactively.

Security rule: actor_id & strategy_id are NEVER tool parameters — they come from
RunnableConfig (set by the handler from the authenticated user) so the LLM can't read another user's memory.

Standard namespace: /strategies/{MEMORY_STRATEGY_ID}/actors/{user_id}
(must match the namespaceTemplate used when creating the memory with /agentbase-memory).
"""

from __future__ import annotations

import logging
from typing import Protocol

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from app.config import Settings
from app.observability import tracing

log = logging.getLogger(__name__)


class LongTermMemory(Protocol):
    def namespace(self, user_id: str) -> str: ...
    async def search(self, user_id: str, query: str, limit: int) -> list[str]: ...
    async def save(self, user_id: str, fact: str) -> None: ...


class AgentBaseLTM:
    def __init__(self, settings: Settings):
        from app.memory.short_term import memory_client

        self._client = memory_client(settings)
        self._max_query = settings.memory_query_max_chars
        self._memory_id = settings.memory_id
        self._strategy_id = settings.memory_strategy_id
        self._min_score = settings.ltm_min_score

    def namespace(self, user_id: str) -> str:
        return f"/strategies/{self._strategy_id}/actors/{user_id}"

    async def search(self, user_id: str, query: str, limit: int) -> list[str]:
        from greennode_agentbase.memory.models import MemoryRecordSearchRequest

        results = await self._client.search_memory_records_async(
            id=self._memory_id,
            namespace=self.namespace(user_id),
            # The API limits query to ≤ 1000 chars (400 if exceeded) — keep the end (usually the question)
            request=MemoryRecordSearchRequest(query=query[-self._max_query :], limit=limit),
        )
        facts = []
        for r in results or []:
            # SDK 1.0.x returns list[dict] {id, memory, score, created_at}; objects are tolerated too
            memory = r.get("memory") if isinstance(r, dict) else getattr(r, "memory", None)
            score = (r.get("score") if isinstance(r, dict) else getattr(r, "score", None)) or 0.0
            if memory and (self._min_score is None or score >= self._min_score):
                facts.append(memory)
        return facts

    async def save(self, user_id: str, fact: str) -> None:
        from greennode_agentbase.memory.models import MemoryRecordInsertDirectlyRequest

        # SDK 1.0.x: request MUST be a MemoryRecordInsertDirectlyRequest (passing a list => TypeError)
        await self._client.insert_memory_records_directly_async(
            id=self._memory_id,
            namespace=self.namespace(user_id),
            request=MemoryRecordInsertDirectlyRequest(memory_records=[fact]),
        )


class InMemoryLTM:
    """Local version for tests: simple keyword matching."""

    def __init__(self) -> None:
        self._facts: dict[str, list[str]] = {}

    def namespace(self, user_id: str) -> str:
        return f"local/{user_id}"

    async def search(self, user_id: str, query: str, limit: int) -> list[str]:
        words = {w for w in query.lower().split() if len(w) > 2}
        facts = self._facts.get(user_id, [])
        scored = [(sum(w in f.lower() for w in words), f) for f in facts]
        return [f for score, f in sorted(scored, reverse=True) if score > 0][:limit]

    async def save(self, user_id: str, fact: str) -> None:
        self._facts.setdefault(user_id, []).append(fact)


_ltm: LongTermMemory | None = None


def get_ltm(settings: Settings) -> LongTermMemory | None:
    global _ltm
    if not settings.ltm_enabled:
        return None
    if _ltm is None:
        _ltm = AgentBaseLTM(settings) if settings.memory_backend == "agentbase" else InMemoryLTM()
    return _ltm


def _user_id(config: RunnableConfig) -> str:
    user_id = (config.get("configurable") or {}).get("actor_id")
    if not user_id:
        raise RuntimeError("actor_id missing in RunnableConfig")
    return user_id


def build_memory_tools(ltm: LongTermMemory, settings: Settings) -> list[BaseTool]:
    @tool
    async def remember(fact: str, config: RunnableConfig) -> str:
        """Save a durable fact about the user (preferences, context, decisions) for use in
        later conversations. Only save information the user shared voluntarily.

        Args:
            fact: A short, self-contained sentence, e.g. "The user prefers answers in Vietnamese".
        """
        user_id = _user_id(config)
        with tracing.step("memory.save", input={"namespace": ltm.namespace(user_id), "fact": fact}):
            await ltm.save(user_id, fact)
        return f"Remembered: {fact}"

    @tool
    async def recall_memory(query: str, config: RunnableConfig) -> str:
        """Search long-term memory for information about the current user.

        Args:
            query: A natural-language query.
        """
        user_id = _user_id(config)
        with tracing.step(
            "memory.recall",
            input={
                "query": query,
                "namespace": ltm.namespace(user_id),
                "limit": settings.ltm_recall_limit,
                "trigger": "tool",
            },
        ) as st:
            facts = await ltm.search(user_id, query, settings.ltm_recall_limit)
            st.set(output={"facts": facts, "count": len(facts)})
        return "\n".join(f"- {f}" for f in facts) if facts else "No relevant information."

    return [remember, recall_memory]


async def auto_recall(ltm: LongTermMemory, user_id: str, query: str, limit: int) -> list[str]:
    """Used in the recall node; memory errors must not break the chat turn."""
    if not query.strip():
        return []
    with tracing.step(
        "memory.recall",
        input={
            "query": query,
            "namespace": ltm.namespace(user_id),
            "limit": limit,
            "trigger": "auto",
        },
    ) as st:
        try:
            facts = await ltm.search(user_id, query, limit)
        except Exception as e:  # noqa: BLE001
            log.warning("long-term recall failed", exc_info=True)
            st.set(
                output={"facts": [], "count": 0},
                level="WARNING",
                status_message=f"{type(e).__name__}: {e}",
            )
            return []
        st.set(output={"facts": facts, "count": len(facts)})
        return facts
