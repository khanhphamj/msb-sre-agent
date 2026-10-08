"""Inbound auth — authenticates the agent's CALLER (frontend / other services).

Principles:
- The user_id used for memory/identity MUST come from the verified token; don't trust the
  X-GreenNode-AgentBase-User-Id header sent by the client. If the client sends it and it differs from `sub` => 403.
- After verification, override user_id in GreenNodeAgentBaseContext so @requires_api_key /
  @requires_access_token (USER_FEDERATION) use the right user.
- AUTH_MODE=none only runs with APP_ENV=local (enforced in config).
- The AgentBase Runtime endpoint is public, with NO platform auth layer (verified) ⇒ jwt or
  api_key is mandatory outside local.
- api_key: only for trusted callers (server/BFF/job/test) — NEVER embed the key in a mobile app. The
  User-Id header is REQUIRED (no shared bucket) — the trusted caller is responsible for its value.
  Env stores only the key's SHA-256.
- Every user_id (header or JWT sub) goes through validate_user_id() before touching memory.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import jwt
from greennode_agentbase import GreenNodeAgentBaseContext, RequestContext
from greennode_agentbase.exceptions import GreenNodeRequestError

from app.config import Settings

LOCAL_USER_ID = "local-user"
# user_id is put into memory paths (/actors/{user_id}, /actors/{user}/sessions/...) ⇒ only safe
# characters allowed; "/", "..", whitespace... are banned to prevent namespace injection / cross-user reads.
USER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@+=-]{0,127}$")
# session_id/contextId goes into Memory paths (/actors/{u}/sessions/{s}/events) — the SDK does NOT encode it, httpx
# normalizes "../" ⇒ "../../victim/sessions/x" reads/writes another user's history (confirmed). Disallow
# "_" (the bridge cache key is f"{thread}_{actor}"), ":" (in-memory key), ".", "/", "%".
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,127}$")


def validate_user_id(user_id: str) -> str:
    if not USER_ID_RE.fullmatch(user_id) or ".." in user_id:
        raise GreenNodeRequestError("Invalid user id format", status_code=400)
    return user_id


def validate_session_id(session_id: str | None) -> str:
    if not session_id or not SESSION_ID_RE.fullmatch(session_id):
        raise GreenNodeRequestError(
            "Invalid session id (letters, digits, '-' only; max 128 chars — use a UUID)",
            status_code=400,
        )
    return session_id


def user_id_from_subject(subject: str, issuer: str | None) -> str:
    """An IdP `sub` may contain path-unsafe characters (Auth0 `auth0|abc`, ...) ⇒ map it stably
    to `u-<sha256(iss|sub)>` instead of rejecting."""
    if USER_ID_RE.fullmatch(subject) and ".." not in subject:
        return subject
    return "u-" + hashlib.sha256(f"{issuer or ''}|{subject}".encode()).hexdigest()[:40]


@dataclass(frozen=True)
class Principal:
    user_id: str
    token: str | None = None
    claims: dict[str, Any] = field(default_factory=dict)


def _unauthorized(msg: str, status: int = 401) -> GreenNodeRequestError:
    return GreenNodeRequestError(msg, status_code=status)


@lru_cache(maxsize=4)
def _jwks_client(url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(url, cache_keys=True, lifespan=3600)


def _header(context: RequestContext, name: str) -> str | None:
    headers = context.request_headers or {}
    lowered = {k.lower(): v for k, v in headers.items()}
    return lowered.get(name.lower())


def _extract_bearer(raw: str | None) -> str | None:
    if not raw:
        return None
    scheme, _, token = raw.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token else raw.strip()


def verify_jwt(token: str, settings: Settings) -> dict[str, Any]:
    try:
        key = _jwks_client(settings.auth_jwks_url).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            key.key,
            algorithms=settings.auth_algorithms,
            audience=settings.auth_audience or None,
            issuer=settings.auth_issuer or None,
            options={"require": ["exp", "iat"], "verify_aud": bool(settings.auth_audience)},
            leeway=30,
        )
    except jwt.PyJWKClientConnectionError as e:
        # IdP/JWKS unreachable ⇒ 503 (not 401 — avoids mass client logouts)
        raise _unauthorized("Identity provider unavailable", status=503) from e
    except jwt.PyJWTError as e:
        raise _unauthorized(f"Invalid token: {type(e).__name__}") from e
    # Only access tokens are accepted (Cognito-style IdPs mark ID tokens with token_use=id)
    if claims.get("token_use", "access") != "access":
        raise _unauthorized("Invalid token: not an access token")
    if settings.auth_allowed_client_ids:
        client = claims.get("azp") or claims.get("client_id")
        if client not in settings.auth_allowed_client_ids:
            raise _unauthorized("Invalid token: client not allowed")
    return claims


def _jwt_token(context: RequestContext, settings: Settings) -> str:
    token = _extract_bearer(_header(context, settings.auth_token_header))
    if not token:
        raise _unauthorized(f"Missing bearer token in header {settings.auth_token_header}")
    return token


def _principal(context: RequestContext, settings: Settings, claims: dict | None, token: str | None):
    claimed_user = context.user_id
    if settings.auth_mode == "none":
        principal = Principal(user_id=claimed_user or LOCAL_USER_ID)
    elif settings.auth_mode == "api_key":
        key = _header(context, settings.auth_api_key_header)
        if not key:
            raise _unauthorized(f"Missing API key in header {settings.auth_api_key_header}")
        digest = hashlib.sha256(key.strip().encode()).hexdigest()
        if not any(hmac.compare_digest(digest, h.lower()) for h in settings.auth_api_key_sha256):
            raise _unauthorized("Invalid API key")
        if not claimed_user:
            raise GreenNodeRequestError(
                "Header X-GreenNode-AgentBase-User-Id is required in api_key mode "
                "(memory is isolated per user)",
                status_code=400,
            )
        principal = Principal(user_id=claimed_user, claims={"auth": "api_key"})
    else:
        raw = (claims or {}).get(settings.auth_user_claim)
        if not raw:
            raise _unauthorized(f"Token missing claim '{settings.auth_user_claim}'")
        user_id = user_id_from_subject(str(raw), (claims or {}).get("iss"))
        if claimed_user and claimed_user not in (str(raw), user_id):
            raise _unauthorized("User-Id header does not match token subject", status=403)
        principal = Principal(user_id=user_id, token=token, claims=claims or {})
    validate_user_id(principal.user_id)
    GreenNodeAgentBaseContext.set_user_id(principal.user_id)
    return principal


def authenticate(context: RequestContext, settings: Settings) -> Principal:
    """Sync version (tests, sync code). The server uses `authenticate_async` so the JWKS fetch doesn't block the loop."""
    if settings.auth_mode == "jwt":
        token = _jwt_token(context, settings)
        return _principal(context, settings, verify_jwt(token, settings), token)
    return _principal(context, settings, None, None)


async def authenticate_async(context: RequestContext, settings: Settings) -> Principal:
    if settings.auth_mode == "jwt":
        token = _jwt_token(context, settings)
        claims = await asyncio.to_thread(verify_jwt, token, settings)  # JWKS fetch is sync I/O
        return _principal(context, settings, claims, token)
    return _principal(context, settings, None, None)


async def prefetch_jwks(settings: Settings) -> None:
    """Called at startup so the first request doesn't wait for the JWKS download (errors are only logged, not blocking)."""
    if settings.auth_mode == "jwt" and settings.auth_jwks_url:
        try:
            await asyncio.to_thread(_jwks_client(settings.auth_jwks_url).get_jwk_set)
        except Exception:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).warning("JWKS prefetch failed", exc_info=True)
