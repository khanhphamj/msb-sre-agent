"""Tools written directly in the agent (in-process).

Rules:
- Only put small, pure-logic tools here that need not be shared with other agents. Shared tools
  or tools calling internal systems => write an MCP server (/agentbase-build-mcp-server) and connect via
  AgentBase MCP Gateway (Connector + Policy Group).
- POC when the internal system has no API yet: mock adapter in-process here, behind a setting BLOCKED in prod;
  keep the tool name so moving to MCP later needs no change to HITL_TOOLS / eval expected_tools.
- The docstring is the "prompt" for the LLM: describe when to use it and what the parameters mean.
- User identity: NEVER take it from tool parameters. Declare a `config: RunnableConfig` parameter (LangChain
  injects it, the LLM doesn't see it) then read:
      user_id = config["configurable"]["actor_id"]           # = authenticated user
      claims  = config["configurable"].get("user_claims", {})  # only claims in AUTH_FORWARD_CLAIMS
- External service secrets: store them in AgentBase Identity and inject with the helpers in app/identity.py
  (agent_api_key / agent_access_token / user_access_token / user_api_key — /agentbase-build-identity), never
  in the prod .env and never as a parameter of the @tool itself.
"""

from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from app.config import get_settings
from app.observability import tracing


@tool
def get_current_time(timezone: str = "Asia/Ho_Chi_Minh") -> str:
    """Get the current date and time in an IANA timezone (default Asia/Ho_Chi_Minh)."""
    return datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- knowledge (RAG POC)
# Internal documents (.md/.txt) in KNOWLEDGE_DIR (default app/knowledge/, kept in the image
# via an exception in .dockerignore). Simple keyword search — enough for a POC / a few dozen documents.
# Prod (many documents, needs semantic search, frequent updates): move to an MCP server with a
# vector store, KEEP the tool name `search_knowledge` and the citation format.
_WORD_RE = re.compile(r"\w+", re.UNICODE)


@lru_cache(maxsize=1)
def _knowledge_chunks(directory: str) -> list[tuple[str, str, str]]:
    """(file, heading, passage) — split on markdown headings / blank lines."""
    chunks: list[tuple[str, str, str]] = []
    base = Path(directory)
    for f in sorted([*base.rglob("*.md"), *base.rglob("*.txt")]):
        heading = f.stem
        for block in re.split(r"\n\s*\n", f.read_text(encoding="utf-8")):
            block = block.strip()
            if not block:
                continue
            if block.startswith("#"):
                heading = block.lstrip("#").strip().splitlines()[0]
            chunks.append((str(f.relative_to(base)), heading, block))
    return chunks


@tool
def search_knowledge(query: str) -> str:
    """Search internal documents (policies, procedures, FAQ) for passages relevant to the question.
    Always use this tool before answering questions about policies/regulations; cite [Source: ...].

    Args:
        query: The question or keywords to look up.
    """
    s = get_settings()
    words = {w for w in _WORD_RE.findall(query.lower()) if len(w) > 1}
    with tracing.step("knowledge.search", input={"query": query}) as st:
        scored = []
        for file, heading, text in _knowledge_chunks(s.knowledge_dir):
            low = text.lower()
            score = sum(low.count(w) for w in words)
            if score:
                scored.append((score, file, heading, text))
        top = sorted(scored, key=lambda x: -x[0])[: s.knowledge_top_k]
        st.set(output={"hits": [(f, h, sc) for sc, f, h, _ in top]})
    if not top:
        return (
            "Not found in internal documents. Tell the user this information is not available yet."
        )
    return "\n\n".join(
        f"[Source: {f} — {h}]\n{t[: s.knowledge_snippet_chars]}" for _, f, h, t in top
    )


# --------------------------------------------------------------------------- per-user tool example
@tool
def whoami(config: RunnableConfig) -> str:
    """Tell who the current user is according to the system (use when the user asks about their account)."""
    conf = config.get("configurable") or {}
    claims = conf.get("user_claims") or {}
    return f"user_id={conf.get('actor_id')} claims={sorted(claims)}"


# Example tool calling an external service with an API key stored in AgentBase Identity
# (decorate the INNER function so the key never appears in the tool schema — /agentbase-build-identity):
#
# from app.identity import agent_api_key
#
# @agent_api_key("weather-api-key")
# async def _fetch_weather(city: str, *, api_key: str) -> dict: ...
#
# @tool
# async def get_weather(city: str) -> str:
#     """Current weather for a city."""
#     return str(await _fetch_weather(city))


def get_local_tools() -> list[BaseTool]:
    # msb-sre-agent has no local tools (Agent Spec §5): every tool comes from the MCP Gateway.
    # get_current_time / whoami above stay as examples only.
    tools: list[BaseTool] = []
    if _knowledge_chunks(get_settings().knowledge_dir):  # only enabled when documents exist
        tools.append(search_knowledge)
    return tools
