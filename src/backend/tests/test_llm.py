"""Capability tiers + task → tier table + fallback + adaptive routing."""

from __future__ import annotations

import httpx
import openai
import pytest
from greennode_agentbase import RequestContext
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app import llm, service
from app.config import get_settings


class _Model(BaseChatModel):
    name_: str
    error: str | None = None

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        req = httpx.Request("POST", "http://x")
        if self.error == "connection":
            raise openai.APIConnectionError(request=req)
        if self.error == "rate":
            raise openai.RateLimitError("429", response=httpx.Response(429, request=req), body=None)
        if self.error == "bad_request":
            raise openai.BadRequestError(
                "400", response=httpx.Response(400, request=req), body=None
            )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(self.name_))])

    def bind_tools(self, tools, **kw):  # type: ignore[override]
        return self


@pytest.fixture
def models(monkeypatch):
    registry: dict[str, _Model] = {}
    monkeypatch.setattr(
        llm, "_chat", lambda model, tier, role: registry.setdefault(model, _Model(name_=model))
    )
    return registry


def _env(monkeypatch, **kv):
    for k, v in kv.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()


def test_capability_tiers_and_defaults(monkeypatch):
    _env(monkeypatch, LLM_MODEL="L", LLM_MODEL_REASONING="R", LLM_MODEL_SMALL="S")
    assert [llm.model_for(t) for t in ("reasoning", "large", "small")] == ["R", "L", "S"]
    _env(monkeypatch, LLM_MODEL_REASONING="", LLM_MODEL_SMALL="")
    assert llm.model_for("reasoning") == llm.model_for("small") == "L"  # empty tier ⇒ large


def test_task_to_tier_mapping_and_override(monkeypatch):
    _env(monkeypatch, LLM_TASK_TIERS='{"agent": "reasoning", "planner": "reasoning"}')
    assert llm.tier_for("agent") == "reasoning"  # override
    assert llm.tier_for("summarize") == "small"  # default
    assert llm.tier_for("eval_judge") == "reasoning"
    assert llm.tier_for("planner") == "reasoning"  # new task added by the project
    assert llm.tier_for("unknown") == "large"
    _env(monkeypatch, LLM_TASK_TIERS='{"agent": "huge"}')
    with pytest.raises(ValueError):
        llm.tier_for("agent")


def test_fallback_chain_per_tier(monkeypatch):
    _env(
        monkeypatch,
        LLM_MODEL="L",
        LLM_FALLBACK_MODELS='["b1","L","b2","b1"]',
        LLM_TIER_FALLBACKS='{"small": ["s-backup"]}',
    )
    assert llm.fallbacks_for("large") == ["b1", "b2"]
    assert llm.fallbacks_for("small") == ["s-backup"]


@pytest.mark.parametrize("err", ["connection", "rate"])
async def test_falls_back_on_infra_errors(monkeypatch, models, err):
    _env(monkeypatch, LLM_MODEL="primary", LLM_FALLBACK_MODELS='["backup"]')
    models["primary"] = _Model(name_="primary", error=err)
    assert (await llm.get_llm("agent", tools=[]).ainvoke("hi")).content == "backup"


async def test_no_fallback_on_bad_request(monkeypatch, models):
    _env(monkeypatch, LLM_MODEL="primary", LLM_FALLBACK_MODELS='["backup"]')
    models["primary"] = _Model(name_="primary", error="bad_request")
    with pytest.raises(openai.BadRequestError):
        await llm.get_llm("agent").ainvoke("hi")


async def test_each_task_uses_its_tier_model(monkeypatch, models):
    _env(
        monkeypatch,
        LLM_MODEL="L",
        LLM_MODEL_REASONING="R",
        LLM_MODEL_SMALL="S",
        LLM_FALLBACK_MODELS="[]",
    )
    assert (await llm.get_llm("summarize").ainvoke("x")).content == "S"
    assert (await llm.get_llm("agent").ainvoke("x")).content == "L"
    assert (await llm.get_llm("eval_judge").ainvoke("x")).content == "R"


@pytest.mark.parametrize(
    ("verdict", "expected_task"),
    [
        ("simple", "agent_simple"),
        ("standard", "agent"),
        ("complex", "agent_complex"),
        ("junk", "agent"),
    ],  # parse error ⇒ standard
)
async def test_adaptive_routing_picks_task(monkeypatch, verdict, expected_task):
    _env(monkeypatch, LLM_ADAPTIVE_ROUTING="true")
    used: list[str] = []

    class _Fake(BaseChatModel):
        task: str

        @property
        def _llm_type(self):
            return "fake"

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            used.append(self.task)
            text = f'{{"complexity": "{verdict}"}}' if self.task == "router" else "ok"
            return ChatResult(generations=[ChatGeneration(message=AIMessage(text))])

        def bind_tools(self, tools, **kw):  # type: ignore[override]
            return self

    fake = lambda task="agent", tools=None: _Fake(task=task)  # noqa: E731
    monkeypatch.setattr("app.graph.builder.get_llm", fake)
    monkeypatch.setattr("app.llm.routing.get_llm", fake)
    await service.handle(
        {"message": "a question"},
        RequestContext(
            session_id=f"r-{expected_task.replace('_', '-')}-{len(verdict)}", user_id="u"
        ),
    )
    assert used == ["router", expected_task]


async def test_reflection_skipped_for_simple_questions(monkeypatch):
    _env(monkeypatch, LLM_ADAPTIVE_ROUTING="true", REFLECTION_ENABLED="true")
    used: list[str] = []

    class _Fake(BaseChatModel):
        task: str

        @property
        def _llm_type(self):
            return "fake"

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            used.append(self.task)
            text = '{"complexity": "simple"}' if self.task == "router" else "hello there"
            return ChatResult(generations=[ChatGeneration(message=AIMessage(text))])

        def bind_tools(self, tools, **kw):  # type: ignore[override]
            return self

    fake = lambda task="agent", tools=None: _Fake(task=task)  # noqa: E731
    for mod in ("app.graph.builder", "app.llm.routing", "app.reflection"):
        monkeypatch.setattr(f"{mod}.get_llm", fake)
    await service.handle({"message": "hi"}, RequestContext(session_id="skip-refl", user_id="u"))
    assert used == ["router", "agent_simple"]  # no judge


def test_primary_switches_fast_fallbacks_keep_backoff(monkeypatch):
    llm._chat.cache_clear()
    _env(monkeypatch, LLM_MODEL="L", LLM_MAX_RETRIES="2", LLM_FALLBACK_MODELS='["b1"]')
    assert llm._chat("L", "large", "primary").max_retries == 0
    # 429 is account-wide: the fallback must still back off and retry
    assert llm._chat("b1", "large", "fallback").max_retries == 2
    llm._chat.cache_clear()
    _env(monkeypatch, LLM_FALLBACK_MODELS="[]")
    assert llm._chat("L", "large", "primary").max_retries == 2
    llm._chat.cache_clear()


def test_warns_when_fallback_chain_exceeds_request_timeout(monkeypatch, caplog):
    llm._warn_if_over_budget.cache_clear()
    _env(
        monkeypatch,
        LLM_MODEL="L",
        LLM_TIMEOUT_S="60",
        REQUEST_TIMEOUT_S="180",
        LLM_TIER_FALLBACKS='{"reasoning": ["r2"]}',
    )
    llm._warn_if_over_budget("reasoning")  # (1 + 1 × 3) attempts × 120s > 180s
    llm._warn_if_over_budget("large")  # no fallback: 1 + 2 retries = 3 × 60s = 180s ⇒ ok
    assert "tier reasoning" in caplog.text and "tier large" not in caplog.text
    llm._warn_if_over_budget.cache_clear()
