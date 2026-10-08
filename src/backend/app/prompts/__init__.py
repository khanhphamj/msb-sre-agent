"""System prompt — Langfuse Prompt Management first, local file as fallback.

- LANGFUSE_PROMPT_NAME empty => use prompts/system.md.
- Name set => fetch the prompt by label (LANGFUSE_PROMPT_LABEL, default "production"),
  SDK caches by TTL; connection lost => local file fallback. The prompt client is attached to the generation
  so Langfuse reports latency/cost/score per prompt version.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from app.observability.tracing import get_client

log = logging.getLogger(__name__)
_LOCAL = (Path(__file__).parent / "system.md").read_text(encoding="utf-8")


def get_system_prompt() -> tuple[str, Any | None]:
    """Returns (text, langfuse_prompt_client | None)."""
    name = os.getenv("LANGFUSE_PROMPT_NAME", "")
    client = get_client()
    if not name or client is None:
        return _LOCAL, None
    label = os.getenv("LANGFUSE_PROMPT_LABEL") or "production"
    try:
        prompt = client.get_prompt(
            name,
            label=label,
            type="text",
            fallback=_LOCAL,
            cache_ttl_seconds=int(os.getenv("LANGFUSE_PROMPT_CACHE_TTL", "300")),
        )
        return prompt.compile(), (None if getattr(prompt, "is_fallback", False) else prompt)
    except Exception:  # noqa: BLE001
        log.warning("Langfuse prompt '%s' unavailable, using local file", name, exc_info=True)
        return _LOCAL, None
