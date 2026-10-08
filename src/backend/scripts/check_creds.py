"""Check the credentials this agent needs. Prints only OK/FAIL and non-secret identifiers — never a secret.

    make check-creds        (= cd src/backend && uv run python -m scripts.check_creds)

A CLI tool, not app runtime code — hence it lives outside app/ and may read the environment directly.

The USER fills these files in their editor (never paste secrets into a chat with a coding agent):
    .greennode.json  IAM service account {"client_id", "client_secret"}  — /agentbase (IAM setup)
    .env             LLM_API_KEY, LLM_MODEL                              — /agentbase-llm
                     LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY (optional) — Langfuse project settings
Exit code 1 if a required credential is missing or rejected.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

import httpx
from dotenv import dotenv_values

from app.config import GREENNODE_LLM_BASE_URL, get_settings, normalize_iam_env

Result = tuple[str, str]  # (status: OK | FAIL | SKIP, detail)


def _env() -> dict[str, str]:
    """`.env` overlaid by the process env — read directly so a check still runs when Settings is invalid."""
    return {k: v for k, v in {**dotenv_values(".env"), **os.environ}.items() if v}


def check_iam() -> Result:
    normalize_iam_env()
    if not (os.getenv("GREENNODE_CLIENT_ID") and os.getenv("GREENNODE_CLIENT_SECRET")):
        return (
            "FAIL",
            "not configured — copy .greennode.json.example to .greennode.json and fill it in",
        )
    from app.tools.mcp import IAMBearerAuth

    auth = IAMBearerAuth()
    try:
        auth.refresh()
    except httpx.HTTPStatusError as e:
        return (
            "FAIL",
            f"rejected by IAM (HTTP {e.response.status_code}) — wrong client_id/secret pair?",
        )
    except httpx.HTTPError as e:
        return "FAIL", f"IAM unreachable ({type(e).__name__})"
    return "OK", f"token issued · principal {auth.principal} (use it in Policy Group)"


def check_agent_identity() -> Result:
    """Read-only. Only needed when tools use AgentBase Identity providers (app/identity.py)."""
    from greennode_agentbase.core.config import get_config_value

    name = get_config_value("GREENNODE_AGENT_IDENTITY")
    if not name:
        return "SKIP", (
            'not set — needed only for Identity providers; put "agent_identity" in .greennode.json '
            "(otherwise the SDK would create a NEW identity on first use)"
        )
    if name.startswith("<"):
        return (
            "FAIL",
            "agent_identity still holds the .example placeholder — set the real name or remove the key",
        )
    return "OK", f"agent identity '{name}'"


def check_llm(env: dict[str, str]) -> Result:
    key, model = env.get("LLM_API_KEY"), env.get("LLM_MODEL")
    if not (key and model):
        return "FAIL", "LLM_API_KEY / LLM_MODEL not set in .env (get them with /agentbase-llm)"
    base = env.get("LLM_BASE_URL", GREENNODE_LLM_BASE_URL).rstrip("/")
    try:  # 1-token completion: proves key AND model path in one call
        r = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            timeout=30,
        )
    except httpx.HTTPError as e:
        return "FAIL", f"LLM endpoint unreachable ({type(e).__name__})"
    if r.status_code in (401, 403):
        return "FAIL", f"LLM_API_KEY rejected (HTTP {r.status_code})"
    if r.status_code >= 400:
        return (
            "FAIL",
            f"model {model!r} not usable (HTTP {r.status_code}) — use the model `path` on AIP",
        )
    return "OK", f"key accepted · model {model}"


def check_langfuse(env: dict[str, str]) -> Result:
    pk, sk = env.get("LANGFUSE_PUBLIC_KEY"), env.get("LANGFUSE_SECRET_KEY")
    if not (pk and sk):
        return "SKIP", "LANGFUSE_* not set — tracing disabled (optional)"
    base = env.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip("/")
    try:
        r = httpx.get(f"{base}/api/public/projects", auth=(pk, sk), timeout=15)
    except httpx.HTTPError as e:
        return "FAIL", f"Langfuse unreachable ({type(e).__name__})"
    if r.status_code != 200:
        return "FAIL", f"Langfuse keys rejected (HTTP {r.status_code})"
    return "OK", f"keys accepted · {base}"


def check_config() -> Result:
    try:
        s = get_settings()
    except (
        ValueError
    ) as e:  # pydantic ValidationError: report the validator's message, not the URL footer
        errors = getattr(e, "errors", None)
        return "FAIL", errors()[0]["msg"] if callable(errors) else str(e)
    return "OK", f"APP_ENV={s.app_env} AUTH_MODE={s.auth_mode} MEMORY_BACKEND={s.memory_backend}"


def main() -> int:
    env = _env()
    checks: list[tuple[str, Callable[[], Result]]] = [
        ("IAM (.greennode.json)", check_iam),
        ("Agent identity", check_agent_identity),
        ("LLM (MaaS)", lambda: check_llm(env)),
        ("Langfuse", lambda: check_langfuse(env)),
        ("Settings (.env)", check_config),
    ]
    failed = False
    for name, fn in checks:
        status, detail = fn()
        failed |= status == "FAIL"
        print(f"{status:<4}  {name:<22} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
