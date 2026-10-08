"""Tests run offline: fake LLM, in-memory memory, auth none, no Langfuse."""

from __future__ import annotations

import os

os.environ.update(
    APP_ENV="local",
    LLM_MODEL="fake-model",
    LLM_API_KEY="fake-key",
    MEMORY_BACKEND="inmemory",
    LTM_ENABLED="true",  # tests cover long-term memory (off by default in production)
    LLM_ADAPTIVE_ROUTING="false",
    A2A_ENABLED="false",
    HITL_TOOLS="[]",
    LLM_FALLBACK_MODELS="[]",
    LLM_TIER_FALLBACKS="{}",
    ZALO_ENABLED="false",
    ZALO_ALLOWED_USER_IDS="[]",
    REFLECTION_ENABLED="false",
    AUTH_MODE="none",
    MCP_CONFIG_FILE="__missing__.json",
    LANGFUSE_PUBLIC_KEY="",
    LANGFUSE_SECRET_KEY="",
)

from typing import Any  # noqa: E402

import pytest  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from app.config import get_settings  # noqa: E402


class FakeChatModel(BaseChatModel):
    """Returns the predefined AIMessages in order; bind_tools returns itself."""

    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = []
    idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeChatModel:  # type: ignore[override]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        msg = self.responses[min(self.idx, len(self.responses) - 1)]
        self.idx += 1
        return ChatResult(generations=[ChatGeneration(message=msg.model_copy())])


@pytest.fixture(autouse=True)
def _fresh_settings():
    get_settings.cache_clear()
    import app.memory.long_term as lt
    import app.memory.short_term as st

    st._checkpointer = None
    lt._ltm = None
    yield
    get_settings.cache_clear()


LLM_MODULES = [
    "app.graph.builder",
    "app.memory.compression",
    "app.reflection",
    "evals.evaluators",
    "app.llm.routing",
]


@pytest.fixture
def fake_llm(monkeypatch):
    def install(*responses: AIMessage) -> FakeChatModel:
        model = FakeChatModel(responses=list(responses), calls=[])
        # Patch EVERY module that imports get_llm — add new modules that use the LLM here
        for target in LLM_MODULES:
            try:
                monkeypatch.setattr(f"{target}.get_llm", lambda task="agent", tools=None: model)
            except (ImportError, AttributeError):
                pass
        return model

    return install
