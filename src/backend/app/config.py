"""Centralized configuration — EVERY environment variable goes through here.

Rules:
- Don't read os.environ scattered around the code; import `get_settings()`.
- Variables injected by AgentBase Runtime (GREENNODE_CLIENT_ID, GREENNODE_CLIENT_SECRET,
  GREENNODE_AGENT_IDENTITY, GREENNODE_ENDPOINT_URL) are NOT declared here — the SDK reads them.
- LANGFUSE_* variables are read by the Langfuse SDK; here we only check whether it is enabled.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

GREENNODE_LLM_BASE_URL = "https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1"


class Settings(BaseSettings):
    # env_ignore_empty: a "KEY=" line in .env (e.g. LLM_MAX_TOKENS=) => use the default instead of a parse error
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True
    )

    # --- App ---
    app_env: Literal["local", "dev", "staging", "prod"] = "local"
    agent_name: str = "msb-sre-agent"
    agent_version: str = "0.1.0"
    log_level: str = "INFO"

    # --- LLM (OpenAI-compatible; defaults to GreenNode AI Platform MaaS) ---
    llm_base_url: str = GREENNODE_LLM_BASE_URL
    llm_api_key: str = ""
    llm_model: str = ""  # use the model's `path` field on AIP (not `code`)
    llm_temperature: float = 0.2
    llm_max_tokens: int | None = None
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    # streaming=True so Langfuse records TTFT (completion_start_time); stream_usage for usage/cost
    # when streaming. If the provider returns no usage when streaming => set LLM_STREAM_USAGE=false.
    llm_streaming: bool = True
    llm_stream_usage: bool = True
    # Tiers by model CAPABILITY (empty => large; large empty => LLM_MODEL) — see app/llm/__init__.py
    llm_model_reasoning: str = ""  # strong reasoning: planning, analysis, math/code, hard grading
    llm_model_large: str = ""  # strong general-purpose + tool calling (default = LLM_MODEL)
    llm_model_small: str = ""  # fast/cheap: router, classification, extraction, summarization
    # Task → tier, overrides DEFAULT_TASK_TIERS, e.g. {"agent": "reasoning", "summarize": "small"}
    llm_task_tiers: dict[str, str] = Field(default_factory=dict)
    # Router scores complexity each turn ⇒ agent uses small / large / reasoning (app/llm/routing.py)
    llm_adaptive_routing: bool = False
    # Fallback on infra/model errors: per tier {"large": [...]} or a shared list
    llm_fallback_models: list[str] = Field(default_factory=list)
    llm_tier_fallbacks: dict[str, list[str]] = Field(default_factory=dict)

    # --- Memory ---
    # agentbase: AgentBase Memory (prod); inmemory: local/test runs without the platform
    memory_backend: Literal["agentbase", "inmemory"] = "agentbase"
    memory_id: str = ""
    memory_strategy_id: str = "default"
    # Long-term memory: enable per the Agent Spec (decision-guide §1) — stores personal data long-term
    ltm_enabled: bool = False
    ltm_auto_recall: bool = True
    ltm_recall_limit: int = 5
    # AgentBase Memory: timeout/retry per call (seen in practice: a checkpoint read hung ~7 minutes when the platform
    # was flaky with the default 30s × 5 retries). Search query max 1000 chars (API limit).
    memory_timeout_s: float = 10.0
    memory_max_retries: int = 3  # 429 is common on bursts ⇒ needs a few attempts; per-request timeout is capped separately
    memory_retry_backoff_s: float = 0.3
    memory_query_max_chars: int = 1000
    # The platform limits concurrent Memory requests to 10 per IAM account (429 "Too many concurrent ...
    # Limit: 10") — SHARED across all runtime replicas ⇒ set ≈ 10 / number of replicas.
    memory_max_concurrency: int = 6
    ltm_min_score: float | None = None

    # --- Graph: max agent ⇄ tools rounds per turn (exceeded ⇒ "too many steps" error, no infinite loop)
    max_tool_rounds: int = 8

    # --- Knowledge (simple internal RAG): .md/.txt files in this directory ⇒ search_knowledge tool
    knowledge_dir: str = "app/knowledge"
    knowledge_top_k: int = 3
    knowledge_snippet_chars: int = 600

    # --- Mask PII before sending traces (email, VN phone, CCCD/CMND) — disable only with good reason
    trace_mask_pii: bool = True

    # --- Time limit for the WHOLE request (/invocations, A2A). Exceeded ⇒ 504 / SSE error 504
    request_timeout_s: float = 180.0

    # --- Context compression ---
    context_max_tokens: int = 12_000  # over threshold => summarize older part
    context_keep_last: int = 8  # messages kept verbatim after summarizing
    context_hard_limit_tokens: int = 24_000  # hard cap before calling the LLM

    # --- MCP ---
    mcp_config_file: str = "mcp_servers.json"
    mcp_tool_timeout_s: float = 30.0
    mcp_list_timeout_s: float = 8.0  # tools/list per server (run in parallel)
    mcp_failure_ttl_s: float = 60.0  # failed server ⇒ skipped for this long (negative cache)

    # --- Inbound auth (authenticates the agent's caller) ---
    # jwt: end-user via IdP (OIDC/JWKS) · api_key: trusted caller (server/BFF/test) · none: local only
    # NOTE: the AgentBase Runtime endpoint does NOT authenticate — the agent MUST protect itself.
    auth_mode: Literal["jwt", "api_key", "none"] = "jwt"
    # Header carrying the user JWT. Use "Authorization" if the runtime endpoint doesn't claim it;
    # if Authorization is already used for IAM, switch to a custom header (the SDK only forwards Authorization
    # and headers prefixed X-GreenNode-AgentBase-Custom-).
    auth_token_header: str = "Authorization"
    auth_jwks_url: str = ""
    auth_issuer: str = ""
    auth_audience: str = ""
    auth_user_claim: str = "sub"
    auth_allow_no_audience: bool = False
    # Allowed OAuth clients (`azp` / `client_id` claim), JSON list. REQUIRED with AUTH_ALLOW_NO_AUDIENCE outside
    # local: without `aud`, this is what stops tokens issued to other apps of the same IdP/user pool.
    auth_allowed_client_ids: list[str] = Field(default_factory=list)
    # JWT claims passed to tools via config["configurable"]["user_claims"] (allowlist, JSON
    # list) — e.g. ["employee_id","roles","email"]. Tools do NOT read the token directly.
    auth_forward_claims: list[str] = Field(default_factory=list)
    auth_algorithms: list[str] = Field(default_factory=lambda: ["RS256", "ES256"])
    # api_key: header carrying the key + SHA-256 hex list of valid keys (JSON list, allows rotation)
    auth_api_key_header: str = "X-GreenNode-AgentBase-Custom-Api-Key"
    auth_api_key_sha256: list[str] = Field(default_factory=list)

    # --- AgentBase Identity (outbound credentials, see app/identity.py) ---
    # App page the user lands on after OAuth2 3LO / delegated-key consent. MUST be in the agent identity's
    # allowedReturnUrls (/agentbase-identity). Empty ⇒ per-user (USER_FEDERATION) credentials are disabled.
    identity_callback_url: str = ""

    # --- Human-in-the-loop: globs on tool names that need human approval, JSON list ---
    # e.g. HITL_TOOLS=["gateway_*_delete","gateway_*_create","send_email"]
    hitl_tools: list[str] = Field(default_factory=list)

    # --- Self-evaluation loop (optional) ---
    reflection_enabled: bool = False
    reflection_max_retries: int = 1
    # Skip reflection when adaptive routing rates the message "simple" (greeting, 1 step) — saves 1–2 LLM calls
    reflection_skip_simple: bool = True
    reflection_criteria: str = (
        "On point, accurate to the tool data, nothing made up, complete, in the user's language."
    )

    # --- A2A (Agent2Agent) — see app/a2a/. Server: other agents call this one; client: a2a_agents.json
    a2a_enabled: bool = False
    # Public URL of the agent (Agent Card). Empty => GREENNODE_ENDPOINT_URL (runtime-injected) or localhost
    a2a_public_url: str = ""
    a2a_description: str = "AgentBase agent"
    a2a_skills: list[dict] = Field(
        default_factory=lambda: [
            {"id": "chat", "name": "Chat", "description": "Answers questions in natural language"}
        ]
    )
    a2a_agents_file: str = "a2a_agents.json"
    a2a_timeout_s: float = 120.0

    # --- Zalo Bot channel (app/channels/zalo.py): webhook in, sendMessage out ---
    zalo_enabled: bool = False
    zalo_api_base: str = "https://bot-api.zaloplatforms.com"
    # Access Control (Identity) Static API Key providers holding the bot token `<id>:<secret>` and the webhook
    # secret (the same value as the secret_token given to setWebhook). Neither belongs in .env.
    zalo_token_provider: str = "msb-sre-zalo-bot-token"
    zalo_secret_provider: str = "msb-sre-zalo-webhook-secret"
    zalo_secret_ttl_s: float = (
        300.0  # how long a secret read from Identity is cached in the process
    )
    # Zalo user ids allowed to chat (JSON list). Empty => nobody: every sender is told their own id so an admin
    # can add it.
    zalo_allowed_user_ids: list[str] = Field(default_factory=list)
    zalo_max_workers: int = (
        3  # chats answered at the same time (MaaS allows 10 requests/min per account)
    )
    zalo_max_queue_per_chat: int = 3  # messages that may wait behind the one being answered
    zalo_notice_after_s: float = (
        10.0  # "still working" notice when no answer after this long (0 = never)
    )
    zalo_timezone: str = "Asia/Ho_Chi_Minh"  # chat sessions rotate at local midnight

    # --- Feedback token (prevents posting scores to another user's trace). Empty => derived from
    # GREENNODE_CLIENT_SECRET (runtime-injected, stable across replicas) or LLM_API_KEY.
    feedback_signing_key: str = ""

    # --- CORS (only needed when the frontend runs on web / Expo web) ---
    cors_allow_origins: list[str] = Field(default_factory=list)

    @property
    def is_local(self) -> bool:
        return self.app_env == "local"

    @staticmethod
    def on_runtime() -> bool:
        """True on AgentBase Runtime: the platform injects GREENNODE_AGENT_IDENTITY (+ client id/secret), and the
        image has no .greennode.json (.dockerignore). Locally the IAM pair lives in .greennode.json, so a developer
        who also exports GREENNODE_AGENT_IDENTITY is NOT mistaken for the Runtime."""
        injected = os.getenv("GREENNODE_AGENT_IDENTITY") or os.getenv("GREENNODE_ENDPOINT_URL")
        return bool(injected) and not Path(".greennode.json").is_file()

    @property
    def signing_key(self) -> bytes:
        seed = self.feedback_signing_key or os.getenv("GREENNODE_CLIENT_SECRET") or self.llm_api_key
        return hashlib.sha256(f"feedback:{seed}".encode()).digest()

    @property
    def langfuse_enabled(self) -> bool:
        if os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() == "false":
            return False
        return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

    @model_validator(mode="after")
    def _validate(self) -> Settings:
        # APP_ENV defaults to "local" (dev convenience). A deploy env file that forgets APP_ENV must NOT
        # inherit local relaxations (AUTH_MODE=none ⇒ anyone can impersonate any user via the User-Id header).
        if self.is_local and self.on_runtime():
            raise ValueError(
                "APP_ENV=local is not allowed on AgentBase Runtime (GREENNODE_AGENT_IDENTITY is injected and "
                "there is no .greennode.json). Set APP_ENV=dev|staging|prod in the deploy env file."
            )
        if not self.llm_model or not self.llm_api_key:
            raise ValueError(
                "LLM_MODEL and LLM_API_KEY are required (use /agentbase-llm to get a key)."
            )
        if self.memory_backend == "agentbase" and not self.memory_id:
            raise ValueError(
                "MEMORY_BACKEND=agentbase requires MEMORY_ID (create it with /agentbase-memory)."
            )
        if (
            self.ltm_enabled
            and self.memory_backend == "agentbase"
            and self.memory_strategy_id in ("", "default")
        ):
            raise ValueError(
                "LTM_ENABLED with MEMORY_BACKEND=agentbase requires MEMORY_STRATEGY_ID = the strategy id of the "
                "memory store (see /agentbase-memory). The namespace is /strategies/<id>/actors/<user>, so a wrong "
                "id makes long-term recall silently return nothing."
            )
        if self.memory_backend == "inmemory" and not self.is_local and self.app_env != "dev":
            raise ValueError("MEMORY_BACKEND=inmemory is only allowed with APP_ENV=local|dev.")
        if self.auth_mode == "none" and not self.is_local:
            raise ValueError("AUTH_MODE=none is only allowed when APP_ENV=local.")
        if self.auth_mode == "jwt" and not self.auth_jwks_url:
            raise ValueError("AUTH_MODE=jwt requires AUTH_JWKS_URL.")
        if self.auth_mode == "jwt" and not self.is_local and not self.auth_issuer:
            raise ValueError("AUTH_MODE=jwt outside local requires AUTH_ISSUER.")
        if (
            self.auth_mode == "jwt"
            and not self.is_local
            and not self.auth_audience
            and not self.auth_allow_no_audience
        ):
            raise ValueError(
                "AUTH_MODE=jwt outside local requires AUTH_AUDIENCE (if empty, tokens for other APIs/clients of the "
                "same IdP would be accepted). IdP doesn't issue `aud` (e.g. Cognito access token) ⇒ "
                "AUTH_ALLOW_NO_AUDIENCE=true."
            )
        if (
            self.auth_mode == "jwt"
            and not self.is_local
            and not self.auth_audience
            and not self.auth_allowed_client_ids
        ):
            raise ValueError(
                "AUTH_ALLOW_NO_AUDIENCE outside local requires AUTH_ALLOWED_CLIENT_IDS (JSON list of the app "
                "client ids allowed to call this agent; checked against the `azp`/`client_id` claim)."
            )
        if self.auth_mode == "api_key" and not self.auth_api_key_sha256:
            raise ValueError(
                "AUTH_MODE=api_key requires AUTH_API_KEY_SHA256 (JSON list of SHA-256 hex)."
            )
        if self.zalo_enabled:
            if not self.zalo_api_base.startswith("https://"):
                raise ValueError("ZALO_API_BASE must be an https:// URL.")
            for provider in (self.zalo_token_provider, self.zalo_secret_provider):
                if not re.fullmatch(r"[A-Za-z0-9_-]{3,50}", provider):
                    raise ValueError(
                        f"Invalid Identity provider name {provider!r} (3-50 chars of A-Za-z0-9_-)."
                    )
            try:
                ZoneInfo(self.zalo_timezone)
            except Exception as e:  # noqa: BLE001 — ZoneInfoNotFoundError / ValueError
                raise ValueError(
                    f"ZALO_TIMEZONE {self.zalo_timezone!r} is not an IANA zone."
                ) from e
        if not self.a2a_public_url:
            self.a2a_public_url = os.getenv("GREENNODE_ENDPOINT_URL") or "http://localhost:8080"
        if self.context_keep_last < 2:
            raise ValueError("CONTEXT_KEEP_LAST must be >= 2.")
        return self


def normalize_iam_env() -> None:
    """Ensure the SDK uses EXACTLY ONE IAM credential pair.

    SDK `IAMCredentials` falls back PER FIELD (env first, then .greennode.json). A shell or .env
    that accidentally has another account's GREENNODE_CLIENT_ID/SECRET => mismatched pair => IAM 401 for Memory,
    Identity, MCP Gateway. Rules:
    - `.greennode.json` has a full pair (local dev) => the file is the ONLY source, overriding env (logs WARNING
      if env differs from the file).
    - No file (Runtime — the file is .dockerignored) => keep the platform-injected env; drop a lone variable.
    """
    log = logging.getLogger(__name__)
    file = Path(".greennode.json")
    if file.is_file():
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        cid, secret = data.get("client_id"), data.get("client_secret")
        if cid and secret:
            if (
                os.getenv("GREENNODE_CLIENT_ID", cid) != cid
                or os.getenv("GREENNODE_CLIENT_SECRET", secret) != secret
            ):
                log.warning(
                    "GREENNODE_CLIENT_* in env differs from .greennode.json — using .greennode.json"
                )
            os.environ["GREENNODE_CLIENT_ID"] = cid
            os.environ["GREENNODE_CLIENT_SECRET"] = secret
            return
    has_id = bool(os.getenv("GREENNODE_CLIENT_ID"))
    has_secret = bool(os.getenv("GREENNODE_CLIENT_SECRET"))
    if has_id != has_secret:
        lone = "GREENNODE_CLIENT_ID" if has_id else "GREENNODE_CLIENT_SECRET"
        log.warning("%s set without its pair — ignoring it", lone)
        os.environ.pop(lone, None)


@lru_cache
def get_settings() -> Settings:
    normalize_iam_env()
    return Settings()
