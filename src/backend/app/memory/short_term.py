"""Short-term memory = LangGraph checkpointer (per-session conversation history).

- agentbase: `AgentBaseMemoryEvents` stores checkpoints as events in AgentBase Memory,
  keyed by (thread_id = session_id, actor_id = user_id). Survives restarts/scale-out.
- inmemory: `InMemorySaver` for local/test — lost on restart, do NOT use in prod.

Required config when invoking the graph:
    {"configurable": {"thread_id": <session_id>, "actor_id": <user_id>}}
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from app.config import Settings

_checkpointer: BaseCheckpointSaver | None = None


class _ConcurrencyLimitedClient:
    """MemoryClient proxy: limits concurrent Memory calls within the process.

    Seen in practice: 8 parallel requests ⇒ Memory API returns 429 "Too many concurrent streaming requests
    for this user. Limit: 10" (limit per IAM account, shared across all replicas). The bridge calls SYNC
    functions (create_event, list_events...) in a thread executor, LTM calls ASYNC functions (*_async) ⇒ both share
    one threading.BoundedSemaphore. The async path polls a NON-blocking acquire: it never blocks the event loop and
    is cancellation-safe (REQUEST_TIMEOUT_S cancels requests; acquiring in an executor thread would let the thread
    take a permit after the coroutine was cancelled — a permanent leak until every Memory call hangs).
    """

    def __init__(self, client: Any, limit: int):
        self._client = client
        self._sem = threading.BoundedSemaphore(max(1, limit))

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._client, name)
        if not callable(attr) or name.startswith("_"):
            return attr
        if inspect.iscoroutinefunction(attr):

            async def async_limited(*args: Any, **kwargs: Any) -> Any:
                delay = 0.005
                while not self._sem.acquire(blocking=False):
                    await asyncio.sleep(delay)  # cancelled here ⇒ nothing was acquired
                    delay = min(delay * 2, 0.05)
                try:
                    return await attr(*args, **kwargs)
                finally:
                    self._sem.release()

            return async_limited

        def limited(*args: Any, **kwargs: Any) -> Any:
            with self._sem:
                return attr(*args, **kwargs)

        return limited


_memory_client: Any = None


def memory_client(settings: Settings) -> Any:
    """Shared MemoryClient (1 semaphore/process) with a short timeout + concurrency limit.

    SDK 1.0.x doesn't accept timeout in the constructor; the internal HttpxClient reads `timeout` when it first
    connects ⇒ set it before the first request."""
    global _memory_client
    if _memory_client is None:
        from greennode_agentbase.memory import MemoryClient

        client = MemoryClient()
        http = getattr(client, "_http_client", None)
        if http is not None and hasattr(http, "timeout"):
            http.timeout = settings.memory_timeout_s
        _memory_client = _ConcurrencyLimitedClient(client, settings.memory_max_concurrency)
    return _memory_client


def get_checkpointer(settings: Settings) -> BaseCheckpointSaver:
    global _checkpointer
    if _checkpointer is None:
        if settings.memory_backend == "agentbase":
            from greennode_agent_bridge import AgentBaseMemoryEvents

            _checkpointer = AgentBaseMemoryEvents(
                memory_id=settings.memory_id,
                memory_client=memory_client(settings),
                max_retries=settings.memory_max_retries,  # bridge default 5
                initial_backoff=settings.memory_retry_backoff_s,  # bridge default 0.1s — too short for 429
            )
        else:
            from langgraph.checkpoint.memory import InMemorySaver

            _checkpointer = InMemorySaver()
    return _checkpointer


def thread_config(settings: Settings, *, session_id: str, user_id: str, **extra) -> dict:
    """Standard config for every graph invocation.

    AgentBaseMemoryEvents isolates by (thread_id, actor_id) itself; InMemorySaver keys only by
    thread_id, so the user is prepended so another user reusing the session_id can't read the history.
    """
    from app.auth.inbound import validate_session_id, validate_user_id

    # Last line of defense for EVERY graph call: both ids go into Memory API paths
    validate_user_id(user_id)
    validate_session_id(session_id)
    thread_id = session_id if settings.memory_backend == "agentbase" else f"{user_id}::{session_id}"
    return {"configurable": {"thread_id": thread_id, "actor_id": user_id, **extra}}
