"""LLM factory: capability TIERS + a TASK → tier table + FALLBACK. The ONLY place that creates LLMs.

Standard LLM source: GreenNode AI Platform (MaaS) `https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1`
(OpenAI-compatible). Model = the `path` field on AIP (/agentbase-llm). Choosing a model per tier: skill
agentbase-build-llm.

1) Capability tiers (each tier: 1 primary model + its own fallback chain)
   reasoning : multi-step reasoning, planning, analysis/math/code, hard grading      (LLM_MODEL_REASONING)
   large     : strong general-purpose + reliable tool calling — default agent       (LLM_MODEL_LARGE | LLM_MODEL)
   small     : fast/cheap — router, classification, extraction, summaries, simple Qs (LLM_MODEL_SMALL)
   Empty tier ⇒ use `large`.

2) Task → tier: LLM_TASK_TIERS overrides DEFAULT_TASK_TIERS. Code only calls get_llm("<task>").

3) Fallback: LLM_TIER_FALLBACKS {"large": [...], ...} or LLM_FALLBACK_MODELS (shared). Fall back only on
   INFRA/MODEL errors (connection lost, timeout, 429, 5xx, model removed/not permitted); NOT on request errors.
   Langfuse: failed (ERROR) generation of the primary model → generation of the fallback model.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableBinding
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from app.config import get_settings

log = logging.getLogger(__name__)

Tier = Literal["reasoning", "large", "small"]
TIERS: tuple[Tier, ...] = ("reasoning", "large", "small")

# Template's standard tasks → tier. Add new tasks (e.g. "planner", "sql") here + LLM_TASK_TIERS.
DEFAULT_TASK_TIERS: dict[str, Tier] = {
    "agent": "large",  # main agent, tool calling (when adaptive routing is off)
    "agent_simple": "small",  # adaptive: greetings, single-step questions, simple lookups
    "agent_complex": "reasoning",  # adaptive: multi-step, analysis, comparison, calculation
    "router": "small",  # scores question complexity (adaptive routing)
    "summarize": "small",  # context compression
    "judge": "large",  # in-turn self-eval (reflection) — must be fast
    "eval_judge": "reasoning",  # offline grading — accuracy first
}

# Errors worth retrying on another model. langchain_openai errors (OpenAIRateLimitError, ...) inherit from these.
FALLBACK_ERRORS: tuple[type[BaseException], ...] = (
    openai.APIConnectionError,  # includes APITimeoutError
    openai.RateLimitError,
    openai.InternalServerError,
    openai.NotFoundError,  # model removed / wrong path
    openai.PermissionDeniedError,  # model not enabled for this key
)


def tier_for(task: str) -> Tier:
    s = get_settings()
    tier = s.llm_task_tiers.get(task) or DEFAULT_TASK_TIERS.get(task) or "large"
    if tier not in TIERS:
        raise ValueError(f"LLM_TASK_TIERS[{task}]={tier} is invalid (only {TIERS})")
    return tier  # type: ignore[return-value]


def model_for(tier: Tier) -> str:
    s = get_settings()
    large = s.llm_model_large or s.llm_model
    return {"reasoning": s.llm_model_reasoning, "large": large, "small": s.llm_model_small}[
        tier
    ] or large


def fallbacks_for(tier: Tier) -> list[str]:
    s = get_settings()
    chain = s.llm_tier_fallbacks.get(tier, s.llm_fallback_models)
    primary, out = model_for(tier), []
    for m in chain:
        if m and m != primary and m not in out:
            out.append(m)
    return out


@lru_cache(maxsize=32)
def _chat(model: str, tier: Tier, role: str) -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=model,
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        temperature=s.llm_temperature if tier != "small" else 0,
        max_tokens=s.llm_max_tokens,
        timeout=s.llm_timeout_s * (2 if tier == "reasoning" else 1),  # reasoning models run longer
        # With fallbacks the PRIMARY doesn't retry (switch fast); fallbacks keep retries so a 429 still gets the
        # SDK's backoff — the MaaS rate limit is per account, shared by every model in the chain.
        max_retries=0 if role == "primary" and fallbacks_for(tier) else s.llm_max_retries,
        streaming=s.llm_streaming,
        stream_usage=s.llm_stream_usage,
        tags=[f"llm.{tier}", f"llm.{role}"],
        metadata={"llm_tier": tier, "llm_role": role, "llm_base_url": s.llm_base_url},
    )


@lru_cache(maxsize=8)
def _warn_if_over_budget(tier: Tier) -> None:
    """Worst case = every model in the chain times out once. If that exceeds REQUEST_TIMEOUT_S the request is
    cancelled before the last fallback can answer — the fallback chain is then partly useless."""
    s = get_settings()
    per_attempt = s.llm_timeout_s * (2 if tier == "reasoning" else 1)
    n_fallbacks = len(fallbacks_for(tier))
    # primary: 1 attempt when fallbacks exist; each fallback: 1 + retries
    attempts = 1 + n_fallbacks * (1 + s.llm_max_retries) if n_fallbacks else 1 + s.llm_max_retries
    if per_attempt * attempts > s.request_timeout_s:
        log.warning(
            "LLM tier %s: worst case %d × %.0fs > REQUEST_TIMEOUT_S=%.0fs — lower LLM_TIMEOUT_S or raise "
            "REQUEST_TIMEOUT_S so the last model in the chain can still answer",
            tier,
            attempts,
            per_attempt,
            s.request_timeout_s,
        )


def get_chat_model(task: str = "agent") -> BaseChatModel:
    """PRIMARY model for the task (no fallback)."""
    tier = tier_for(task)
    return _chat(model_for(tier), tier, "primary")


def get_llm(task: str = "agent", tools: list[BaseTool] | None = None) -> Runnable:
    """Runnable for the task: the tier's primary model (+tools) with that tier's fallback chain.

    Names on Langfuse (the trace tree shows which task called which model):
      llm.<task> | <caller run_name>    task chain (metadata llm_task, llm_tier, prompt link)
        <model>                         primary model GENERATION (model, usage, cost, TTFT)
        <model> (fallback)              backup model GENERATION — only when the primary fails
    """
    tier = tier_for(task)
    primary_model = model_for(tier)
    _warn_if_over_budget(tier)

    def _gen(model: str, role: str) -> Runnable:
        r: Runnable = _chat(model, tier, role)
        if tools:
            r = r.bind_tools(tools)
        display = model if role == "primary" else f"{model} (fallback)"
        # RunnableWithFallbacks passes the chain's run_name down to child models (overriding with_config) ⇒ use
        # config_factories (applied LAST) so the generation always carries the model name on Langfuse.
        return RunnableBinding(bound=r, config_factories=[lambda _cfg: {"run_name": display}])

    primary = _gen(primary_model, "primary")
    backups = [_gen(m, "fallback") for m in fallbacks_for(tier)]
    # ALWAYS wrap in 1 chain (even without fallbacks) ⇒ a caller-set run_name only renames the chain,
    # the inner generation always carries the model name; `langfuse_prompt` metadata passed at call time
    # attaches to this chain ⇒ only that call's generation gets linked to the prompt.
    return primary.with_fallbacks(backups, exceptions_to_handle=FALLBACK_ERRORS).with_config(
        run_name=f"llm.{task}", metadata={"llm_task": task, "llm_tier": tier}
    )
