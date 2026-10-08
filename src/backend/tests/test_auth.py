from __future__ import annotations

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from greennode_agentbase import RequestContext
from greennode_agentbase.exceptions import GreenNodeRequestError

from app.auth import inbound
from app.config import get_settings


@pytest.fixture
def jwt_env(monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setenv("AUTH_MODE", "jwt")
    monkeypatch.setenv("AUTH_JWKS_URL", "https://issuer.test/jwks.json")
    monkeypatch.setenv("AUTH_ISSUER", "https://issuer.test")
    monkeypatch.setenv("AUTH_AUDIENCE", "agent-api")
    get_settings.cache_clear()

    class _Key:
        def __init__(self, k):
            self.key = k

    class _Client:
        def get_signing_key_from_jwt(self, token):
            return _Key(key.public_key())

    monkeypatch.setattr(inbound, "_jwks_client", lambda url: _Client())

    def make(sub="user-1", **over):
        now = int(time.time())
        claims = {
            "sub": sub,
            "iss": "https://issuer.test",
            "aud": "agent-api",
            "iat": now,
            "exp": now + 300,
            **over,
        }
        claims = {k: v for k, v in claims.items() if v is not None}  # aud=None ⇒ no `aud` claim
        return jwt.encode(claims, key, algorithm="RS256")

    return make


def _ctx(token=None, user=None):
    headers = {"Authorization": f"Bearer {token}"} if token else None
    return RequestContext(session_id="s", user_id=user, request_headers=headers)


def test_valid_token_sets_user(jwt_env):
    p = inbound.authenticate(_ctx(jwt_env()), get_settings())
    assert p.user_id == "user-1"


def test_missing_token(jwt_env):
    with pytest.raises(GreenNodeRequestError) as e:
        inbound.authenticate(_ctx(), get_settings())
    assert e.value.status_code == 401


def test_expired_and_wrong_audience(jwt_env):
    for token in (jwt_env(exp=int(time.time()) - 100), jwt_env(aud="other")):
        with pytest.raises(GreenNodeRequestError):
            inbound.authenticate(_ctx(token), get_settings())


def test_user_header_spoofing_rejected(jwt_env):
    with pytest.raises(GreenNodeRequestError) as e:
        inbound.authenticate(_ctx(jwt_env(), user="someone-else"), get_settings())
    assert e.value.status_code == 403


@pytest.fixture
def api_key_env(monkeypatch):
    import hashlib

    monkeypatch.setenv("AUTH_MODE", "api_key")
    monkeypatch.setenv("AUTH_API_KEY_SHA256", f'["{hashlib.sha256(b"secret-key-1").hexdigest()}"]')
    get_settings.cache_clear()


def _key_ctx(key=None, user=None):
    headers = {"X-GreenNode-AgentBase-Custom-Api-Key": key} if key else None
    return RequestContext(session_id="s", user_id=user, request_headers=headers)


def test_api_key_valid_uses_caller_user_id(api_key_env):
    assert (
        inbound.authenticate(_key_ctx("secret-key-1", "alice"), get_settings()).user_id == "alice"
    )


def test_api_key_requires_user_id_no_shared_bucket(api_key_env):
    with pytest.raises(GreenNodeRequestError) as e:
        inbound.authenticate(_key_ctx("secret-key-1"), get_settings())
    assert e.value.status_code == 400


def test_api_key_missing_or_wrong(api_key_env):
    for ctx in (_key_ctx(), _key_ctx("wrong")):
        with pytest.raises(GreenNodeRequestError) as e:
            inbound.authenticate(ctx, get_settings())
        assert e.value.status_code == 401


def test_api_key_mode_requires_hashes(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "api_key")
    monkeypatch.delenv("AUTH_API_KEY_SHA256", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="AUTH_API_KEY_SHA256"):
        get_settings()


# --- APP_ENV must never default to local on the Runtime (local allows AUTH_MODE=none ⇒ User-Id spoofing)
def test_local_app_env_refused_on_runtime(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # Runtime image: no .greennode.json
    # Deploy env file that forgot APP_ENV ⇒ default "local"
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.setenv("GREENNODE_AGENT_IDENTITY", "injected-by-runtime")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="APP_ENV=local is not allowed on AgentBase Runtime"):
        get_settings()


def test_local_dev_with_identity_env_still_allowed(monkeypatch, tmp_path):
    """A developer may export GREENNODE_AGENT_IDENTITY locally (Identity decorators); .greennode.json marks local."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".greennode.json").write_text('{"client_id": "c", "client_secret": "s"}')
    monkeypatch.setenv("GREENNODE_AGENT_IDENTITY", "my-agent")
    get_settings.cache_clear()
    assert get_settings().is_local


# --- No-audience IdPs (e.g. Cognito access tokens): the client allowlist replaces `aud`
@pytest.fixture
def no_aud_env(jwt_env, monkeypatch):
    monkeypatch.delenv("AUTH_AUDIENCE", raising=False)
    monkeypatch.setenv("AUTH_ALLOW_NO_AUDIENCE", "true")
    monkeypatch.setenv("AUTH_ALLOWED_CLIENT_IDS", '["app-1"]')
    get_settings.cache_clear()
    return jwt_env


def test_no_audience_accepts_allowed_client(no_aud_env):
    for claim in ("client_id", "azp"):
        token = no_aud_env(aud=None, **{claim: "app-1"}, token_use="access")
        assert inbound.authenticate(_ctx(token), get_settings()).user_id == "user-1"


def test_no_audience_rejects_other_client_and_id_tokens(no_aud_env):
    for token in (
        no_aud_env(aud=None, client_id="other-app"),  # another app of the same user pool
        no_aud_env(aud=None),  # no client claim at all
        no_aud_env(aud=None, client_id="app-1", token_use="id"),  # ID token, not access token
    ):
        with pytest.raises(GreenNodeRequestError) as e:
            inbound.authenticate(_ctx(token), get_settings())
        assert e.value.status_code == 401


def test_no_audience_outside_local_requires_client_allowlist(monkeypatch):
    for k, v in {
        "APP_ENV": "prod",
        "AUTH_MODE": "jwt",
        "AUTH_JWKS_URL": "https://issuer.test/jwks.json",
        "AUTH_ISSUER": "https://issuer.test",
        "AUTH_ALLOW_NO_AUDIENCE": "true",
        "MEMORY_BACKEND": "agentbase",
        "MEMORY_ID": "mem-1",
        "MEMORY_STRATEGY_ID": "strat-1",  # valid memory config: isolate the auth check
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AUTH_AUDIENCE", raising=False)
    monkeypatch.delenv("AUTH_ALLOWED_CLIENT_IDS", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="AUTH_ALLOWED_CLIENT_IDS"):
        get_settings()
