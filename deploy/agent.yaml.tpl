# AgentBase manifest for `grn agentbase deploy up` — rendered in CI with envsubst (.github/workflows/ci.yml).
# Commit THIS template only; the rendered file contains secrets. Every value is quoted so ':' or '#' in a value
# cannot break the YAML (a value containing '"' would — keep secrets URL-safe).
#
# name = shared join key of identity + runtime (3–50 chars, ^[a-zA-Z0-9_-]+$). The identity is ALWAYS created/kept
# with this name — it is the identity your Access Control providers must belong to (/agentbase-build-identity).
name: "${AGENT_NAME}"
description: "${AGENT_NAME} (deployed by CI)"
identity:
  # Add this environment's IDENTITY_CALLBACK_URL here when tools use per-user credentials (3LO / delegated)
  allowedReturnUrls: []
# memory: intentionally omitted — the memory store is created once with /agentbase-memory and referenced by
# MEMORY_ID / MEMORY_STRATEGY_ID below (its namespaceTemplate must be /strategies/{memoryStrategyId}/actors/{actorId}).
runtime:
  image: "${IMAGE}"
  imageAuth: auto # resolve pull credentials from your vCR robot account
  command: ["python", "main.py"] # same as the Dockerfile CMD
  args: []
  flavorId: "${RUNTIME_FLAVOR}"
  # Thresholds must be within 25–75 % (runtime-reference); replicas 1–10
  autoscaling: {minReplicas: 1, maxReplicas: 1, cpuUtilization: 70, memoryUtilization: 70}
  env:
    APP_ENV: "${APP_ENV}" # required — APP_ENV=local is refused on the Runtime
    LLM_API_KEY: "${LLM_API_KEY}"
    LLM_MODEL: "${LLM_MODEL}"
    MEMORY_BACKEND: "agentbase"
    MEMORY_ID: "${MEMORY_ID}"
    MEMORY_STRATEGY_ID: "${MEMORY_STRATEGY_ID}"
    AUTH_MODE: "jwt"
    AUTH_JWKS_URL: "${AUTH_JWKS_URL}"
    AUTH_ISSUER: "${AUTH_ISSUER}"
    AUTH_AUDIENCE: "${AUTH_AUDIENCE}"
    LANGFUSE_PUBLIC_KEY: "${LANGFUSE_PUBLIC_KEY}"
    LANGFUSE_SECRET_KEY: "${LANGFUSE_SECRET_KEY}"
    LANGFUSE_BASE_URL: "${LANGFUSE_BASE_URL}"
    # Add the rest of your .env.<env> here (MCP_GATEWAY_URL, HITL_TOOLS, LTM_ENABLED, IDENTITY_CALLBACK_URL…).
    # Never GREENNODE_CLIENT_ID / _SECRET / GREENNODE_AGENT_IDENTITY — the Runtime injects them.
