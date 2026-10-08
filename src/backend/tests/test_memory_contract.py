"""Contract tests against the greennode-agentbase SDK (real bugs hit on deploy):
- insert-directly must use MemoryRecordInsertDirectlyRequest (passing a list => TypeError in the SDK)
- search returns list[dict] {id, memory, score, ...}
"""

from __future__ import annotations

from greennode_agentbase.memory.models import (
    MemoryRecordInsertDirectlyRequest,
    MemoryRecordSearchRequest,
)

from app.config import get_settings
from app.memory.long_term import AgentBaseLTM


class FakeMemoryClient:
    def __init__(self):
        self.inserted = []

    async def insert_memory_records_directly_async(self, *, id, namespace, request):
        assert isinstance(request, MemoryRecordInsertDirectlyRequest)
        self.inserted.append((namespace, request.memory_records))
        return {}

    async def search_memory_records_async(self, *, id, namespace, request):
        assert isinstance(request, MemoryRecordSearchRequest)
        assert len(request.query) <= 1000  # Memory API limit (400 if exceeded)
        return [
            {"id": "1", "memory": "Likes green tea", "score": 0.71},
            {"id": "2", "memory": "Unrelated", "score": 0.1},
        ]


def _ltm(monkeypatch, min_score=None):
    monkeypatch.setenv("MEMORY_STRATEGY_ID", "ltms-x")
    if min_score is not None:
        monkeypatch.setenv("LTM_MIN_SCORE", str(min_score))
    get_settings.cache_clear()
    ltm = AgentBaseLTM.__new__(AgentBaseLTM)
    s = get_settings()
    ltm._client, ltm._memory_id = FakeMemoryClient(), "mem-1"
    ltm._strategy_id, ltm._min_score = s.memory_strategy_id, s.ltm_min_score
    ltm._max_query = s.memory_query_max_chars
    return ltm


async def test_save_uses_request_model_and_namespace(monkeypatch):
    ltm = _ltm(monkeypatch)
    await ltm.save("u1", "Likes green tea")
    assert ltm._client.inserted == [("/strategies/ltms-x/actors/u1", ["Likes green tea"])]


async def test_search_parses_dicts_and_min_score(monkeypatch):
    assert await _ltm(monkeypatch).search("u1", "what to drink", 5) == [
        "Likes green tea",
        "Unrelated",
    ]
    assert await _ltm(monkeypatch, min_score=0.5).search("u1", "what to drink", 5) == [
        "Likes green tea"
    ]


async def test_long_query_truncated_to_api_limit(monkeypatch):
    ltm = _ltm(monkeypatch)
    assert await ltm.search("u1", "x" * 5000 + " final question", 5)  # no 400


async def test_memory_concurrency_limited():
    """Never exceed MEMORY_MAX_CONCURRENCY concurrent Memory calls (sync + async)."""
    import asyncio
    import threading
    import time

    from app.memory.short_term import _ConcurrencyLimitedClient

    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def track(delta):
        with lock:
            state["now"] += delta
            state["peak"] = max(state["peak"], state["now"])

    class _Raw:
        def list_events(self):
            track(1)
            time.sleep(0.05)
            track(-1)

        async def search_memory_records_async(self):
            track(1)
            await asyncio.sleep(0.05)
            track(-1)

    c = _ConcurrencyLimitedClient(_Raw(), limit=3)
    loop = asyncio.get_running_loop()
    await asyncio.gather(
        *(loop.run_in_executor(None, c.list_events) for _ in range(6)),
        *(c.search_memory_records_async() for _ in range(6)),
    )
    assert state["peak"] <= 3


async def test_cancelled_waiter_does_not_leak_a_permit():
    """REQUEST_TIMEOUT_S cancels requests: a coroutine cancelled while WAITING for a permit must not take
    one later (a leak per timeout ⇒ after MEMORY_MAX_CONCURRENCY timeouts every Memory call hangs)."""
    import asyncio

    from app.memory.short_term import _ConcurrencyLimitedClient

    release = asyncio.Event()

    class _Raw:
        async def search_memory_records_async(self):
            await release.wait()

    c = _ConcurrencyLimitedClient(_Raw(), limit=2)
    holders = [asyncio.create_task(c.search_memory_records_async()) for _ in range(2)]
    await asyncio.sleep(0.02)  # both permits taken
    waiter = asyncio.create_task(c.search_memory_records_async())
    await asyncio.sleep(0.02)
    waiter.cancel()  # e.g. request timeout while waiting
    await asyncio.gather(waiter, return_exceptions=True)
    release.set()
    await asyncio.gather(*holders)
    await asyncio.sleep(0.1)  # let any stray acquirer run
    # Both permits are free again
    assert c._sem.acquire(blocking=False) and c._sem.acquire(blocking=False)
    c._sem.release()
    c._sem.release()


async def test_compression_never_deletes_unsummarized_or_splits_tool_pairs(fake_llm, monkeypatch):
    """Summarizer input over budget ⇒ only the summarized prefix may be removed, and neither the summarizer
    input nor the kept history may start with an orphan ToolMessage. Swept over many message sizes."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from app.config import get_settings
    from app.memory.compression import summarize_if_needed

    monkeypatch.setenv("CONTEXT_MAX_TOKENS", "1000")
    monkeypatch.setenv("CONTEXT_HARD_LIMIT_TOKENS", "2000")
    monkeypatch.setenv("CONTEXT_KEEP_LAST", "2")
    get_settings.cache_clear()

    def call(i):
        return AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": f"t{i}"}], id=f"a{i}")

    checked = 0
    for size in range(1000, 9000, 250):
        msgs = [
            HumanMessage("q1 " * 50, id="h1"),
            call(1),
            ToolMessage("r" * 3000, tool_call_id="t1", id="tm1"),
            AIMessage("x" * size, id="ans1"),
            HumanMessage("q2 " * 300, id="h2"),
            call(2),
            ToolMessage("r" * size, tool_call_id="t2", id="tm2"),
            AIMessage("y" * 1500, id="ans2"),
            HumanMessage("q3", id="h3"),
            AIMessage("ok", id="ans3"),
        ]
        model = fake_llm(AIMessage("- summary"))
        update = await summarize_if_needed(msgs, "", get_settings())
        if update is None:
            continue
        checked += 1
        seen = model.calls[0][
            1:-1
        ]  # minus system prompt and the final "Return the updated summary."
        removed = {m.id for m in update["messages"]}
        assert removed <= {m.id for m in seen}, f"size={size}: removed but never summarized"
        kept = [m for m in msgs if m.id not in removed]
        for part in (seen, kept):
            assert not isinstance(part[0], ToolMessage), f"size={size}: orphan ToolMessage"
            calls = {tc["id"] for m in part if isinstance(m, AIMessage) for tc in m.tool_calls}
            assert all(m.tool_call_id in calls for m in part if isinstance(m, ToolMessage))
    assert checked >= 10  # the sweep really exercised compression


def test_ltm_on_agentbase_requires_real_strategy_id(monkeypatch):
    import pytest

    from app.config import get_settings

    for k, v in {
        "MEMORY_BACKEND": "agentbase",
        "MEMORY_ID": "mem-1",
        "LTM_ENABLED": "true",
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("MEMORY_STRATEGY_ID", "default")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="MEMORY_STRATEGY_ID"):
        get_settings()
    monkeypatch.setenv("MEMORY_STRATEGY_ID", "strat-123")
    get_settings.cache_clear()
    assert get_settings().memory_strategy_id == "strat-123"


def test_huge_user_message_is_truncated_to_budget(monkeypatch):
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.messages.utils import count_tokens_approximately

    from app.config import get_settings
    from app.memory.compression import fit_to_budget

    monkeypatch.setenv("CONTEXT_HARD_LIMIT_TOKENS", "2000")
    get_settings.cache_clear()
    msgs = [SystemMessage("sys"), HumanMessage("START " + "x" * 400_000 + " END")]
    out = fit_to_budget(msgs, get_settings())
    assert count_tokens_approximately(out) <= 2000
    text = out[-1].content
    assert text.startswith("START") and text.endswith("END") and "truncated" in text


def test_user_message_kept_when_ai_part_is_the_problem(monkeypatch):
    """Over budget because of AI text / tool args in the current turn ⇒ do not mangle the user's question."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from app.config import get_settings
    from app.memory.compression import fit_to_budget

    monkeypatch.setenv("CONTEXT_HARD_LIMIT_TOKENS", "2000")
    get_settings.cache_clear()
    question = "q" * 1550
    msgs = [
        SystemMessage("sys"),
        HumanMessage(question),
        AIMessage("", tool_calls=[{"name": "t", "args": {"blob": "a" * 20000}, "id": "t1"}]),
        AIMessage("b" * 6000),
    ]
    out = fit_to_budget(msgs, get_settings())
    assert next(m for m in out if isinstance(m, HumanMessage)).content == question
