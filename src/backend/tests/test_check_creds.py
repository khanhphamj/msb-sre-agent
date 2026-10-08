"""make check-creds: correct verdicts, and it NEVER prints a secret (users run it in chats with coding agents)."""

from __future__ import annotations

import json

import httpx
import pytest

from app.tools.mcp import IAMBearerAuth
from scripts import check_creds

SECRETS = {
    "client_secret": "iam-secret-DO-NOT-PRINT",
    "LLM_API_KEY": "llm-key-DO-NOT-PRINT",
    "LANGFUSE_SECRET_KEY": "lf-secret-DO-NOT-PRINT",
}


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Empty project dir: no developer .greennode.json / .env leaks in, no GREENNODE_* from the shell."""
    monkeypatch.chdir(tmp_path)
    for k in (
        "GREENNODE_CLIENT_ID",
        "GREENNODE_CLIENT_SECRET",
        "GREENNODE_AGENT_IDENTITY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    ):
        monkeypatch.setenv(k, "x")  # registers the original value so teardown restores it…
        monkeypatch.delenv(k)  # …even though normalize_iam_env() writes GREENNODE_* during the test
    import greennode_agentbase.core.config as sdk_config

    monkeypatch.setattr(sdk_config, "_config_cache", None)  # SDK caches .greennode.json per process
    return tmp_path


def _creds_file(path, secret=SECRETS["client_secret"]):
    (path / ".greennode.json").write_text(
        json.dumps({"client_id": "cid-1", "client_secret": secret})
    )


def _fake_http(monkeypatch, llm=200, langfuse=200, iam=200):
    def refresh(self):
        if iam != 200:
            req = httpx.Request("POST", "https://iam.test")
            raise httpx.HTTPStatusError("x", request=req, response=httpx.Response(iam, request=req))
        self.principal = "iam:sa-123"

    monkeypatch.setattr(IAMBearerAuth, "refresh", refresh)
    monkeypatch.setattr(check_creds.httpx, "post", lambda *a, **k: httpx.Response(llm))
    monkeypatch.setattr(check_creds.httpx, "get", lambda *a, **k: httpx.Response(langfuse))


def test_all_ok_shows_principal_and_no_secret(workdir, monkeypatch, capsys):
    _creds_file(workdir)
    monkeypatch.setenv("LLM_API_KEY", SECRETS["LLM_API_KEY"])
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", SECRETS["LANGFUSE_SECRET_KEY"])
    _fake_http(monkeypatch)
    assert check_creds.main() == 0
    out = capsys.readouterr().out
    assert "iam:sa-123" in out
    assert not any(secret in out for secret in SECRETS.values())


def test_missing_iam_file_fails(workdir, monkeypatch, capsys):
    _fake_http(monkeypatch)
    assert check_creds.main() == 1
    assert ".greennode.json.example" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("kw", "expected"),
    [
        ({"iam": 401}, "rejected by IAM"),
        ({"llm": 401}, "LLM_API_KEY rejected"),
        ({"llm": 404}, "not usable"),
    ],
)
def test_rejected_credentials_fail(workdir, monkeypatch, capsys, kw, expected):
    _creds_file(workdir)
    _fake_http(monkeypatch, **kw)
    assert check_creds.main() == 1
    assert expected in capsys.readouterr().out


def test_langfuse_optional(workdir, monkeypatch, capsys):
    _creds_file(workdir)
    _fake_http(monkeypatch)
    assert check_creds.main() == 0
    assert "SKIP  Langfuse" in capsys.readouterr().out


def test_agent_identity_placeholder_fails(workdir, monkeypatch, capsys):
    import greennode_agentbase.core.config as sdk_config

    (workdir / ".greennode.json").write_text(
        json.dumps(
            {"client_id": "c", "client_secret": "s", "agent_identity": "<agent identity name>"}
        )
    )
    monkeypatch.setattr(sdk_config, "_config_cache", None)
    monkeypatch.delenv("GREENNODE_AGENT_IDENTITY", raising=False)
    _fake_http(monkeypatch)
    assert check_creds.main() == 1
    assert "placeholder" in capsys.readouterr().out
