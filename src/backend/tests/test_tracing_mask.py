"""Trace masking: secrets never leave the process; usage numbers and amounts stay readable."""

from __future__ import annotations

import pytest

from app.observability import tracing


@pytest.fixture(autouse=True)
def _pii_on(monkeypatch):
    monkeypatch.setattr(tracing, "_mask_pii", True)


@pytest.mark.parametrize(
    "key",
    [
        "Authorization",
        "X-API-Key",
        "api-key",
        "refresh_token",
        "id_token",
        "client-secret",
        "LANGFUSE_SECRET_KEY",
        "private_key",
        "Cookie",
        "set-cookie",
        "X-GreenNode-AgentBase-Custom-Api-Key",
        "feedback_token",
        "accessToken",
        "refreshToken",
        "clientSecret",
    ],
)
def test_secret_keys_masked(key):
    assert tracing._mask(data={key: "s3cr3t-value"}) == {key: "***"}


@pytest.mark.parametrize(
    "key", ["max_tokens", "prompt_tokens", "tokens_before", "auth_mode", "model"]
)
def test_non_secret_keys_kept(key):
    assert tracing._mask(data={key: 1234}) == {key: 1234}


def test_secret_values_masked_in_text():
    text = (
        "Authorization: Bearer abc.def.ghi Basic dXNlcjpwYXNzMTIz key sk-proj-ABCDEFGHIJKLMNOPQRSTUV "
        "maas vn-FAKEKEY_aaaaBBBBccccDDDDeeeeFFFF0000"
    )
    out = tracing._mask(data=text)
    for leaked in ("abc.def.ghi", "dXNlcjpwYXNzMTIz", "sk-proj-ABCDEF", "vn-FAKEKEY"):
        assert leaked not in out


def test_pii_cards_and_ids_masked_amounts_kept():
    out = tracing._mask(
        data="card 4111 1111 1111 1111, CMND: 123456789, CCCD 012345678901, mail a@b.vn, "
        "amount 150000000 VND, order 1234567890123 (not a valid card)"
    )
    assert "4111" not in out and "123456789," not in out and "012345678901" not in out
    assert "a@b.vn" not in out
    assert "150000000" in out  # 9-digit amount without an ID keyword is kept
    assert "1234567890123" in out  # 13 digits failing Luhn is not a card


def test_email_user_id_hashed_for_trace():
    assert tracing.trace_user_id("alice@corp.vn").startswith("u-")
    assert tracing.trace_user_id("user-123") == "user-123"


@pytest.mark.parametrize("key", ["token_usage", "credential_type", "token_use", "expires_in"])
def test_metadata_keys_not_masked(key):
    assert tracing._mask(data={key: "v"}) == {key: "v"}


def test_prose_and_timestamps_not_masked():
    text = (
        "The basic idea is simple. Basic plan costs 100k. Basic Information here. "
        "Bearer of good news. ts=1728300000002"
    )
    assert tracing._mask(data=text) == text
