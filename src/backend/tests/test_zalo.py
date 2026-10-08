"""Zalo Bot channel — webhook security, allow-list, per-chat ordering, error replies. All offline: a fake Access
Control (Identity), a fake Zalo API (httpx.MockTransport) and a fake agent (`run_chat`)."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest
from greennode_agentbase.exceptions import GreenNodeRequestError
from starlette.applications import Starlette

from app.auth.inbound import validate_session_id, validate_user_id
from app.channels import zalo
from app.config import Settings, get_settings

TOKEN = "123456:BOT-TOKEN-SECRET"
SECRET = "webhook-secret-0123456789"
ALLOWED = "user-allowed-1"
NOW = datetime(2026, 10, 7, 9, 30, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))


# --------------------------------------------------------------------------- fakes
class FakeProviders:
    """Stands in for Access Control: provider name -> value; an unknown provider raises like the SDK would."""

    def __init__(self, values: dict[str, str]):
        self.values = dict(values)
        self.calls: list[str] = []

    async def __call__(self, provider: str) -> str:
        self.calls.append(provider)
        if provider not in self.values:
            raise RuntimeError("provider not found")
        return self.values[provider]


class FakeAgent:
    """Stands in for service.run_chat."""

    def __init__(self, answer: str = "OK", *, delay: float = 0.0, error: Exception | None = None):
        self.answer, self.delay, self.error = answer, delay, error
        self.calls: list[dict] = []
        self.gate: asyncio.Event | None = None

    async def __call__(self, message: str, *, user_id: str, session_id: str) -> dict:
        self.calls.append({"message": message, "user_id": user_id, "session_id": session_id})
        if self.gate is not None:
            await self.gate.wait()
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return {"status": "success", "response": self.answer}


def make_settings(monkeypatch, **env) -> Settings:
    base = {
        "ZALO_ENABLED": "true",
        "ZALO_ALLOWED_USER_IDS": json.dumps([ALLOWED]),
        "ZALO_NOTICE_AFTER_S": "0",
    }
    for key, value in {**base, **env}.items():
        monkeypatch.setenv(key, str(value))
    get_settings.cache_clear()
    return get_settings()


def build(settings, run_chat, *, providers=None, zalo_handler=None):
    sent: list[dict] = []

    def default_handler(request: httpx.Request) -> httpx.Response:
        sent.append({"url": str(request.url), "body": json.loads(request.content)})
        return httpx.Response(200, json={"ok": True, "result": {}})

    providers = providers or FakeProviders(
        {settings.zalo_token_provider: TOKEN, settings.zalo_secret_provider: SECRET}
    )
    cache = zalo.SecretCache(60, fetch=providers)
    client = zalo.ZaloClient(
        settings, cache, transport=httpx.MockTransport(zalo_handler or default_handler)
    )
    ctx = zalo.ZaloContext(
        settings, client=client, secrets_cache=cache, run_chat=run_chat, clock=lambda: NOW
    )
    app = Starlette(routes=zalo.zalo_routes(settings, ctx=ctx))
    return ctx, app, sent


async def post(app, body, *, secret: str | None = SECRET, raw: str | None = None):
    headers = {"Content-Type": "application/json"}
    if secret is not None:
        headers["X-Bot-Api-Secret-Token"] = secret
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.post(
            "/webhook/zalo", content=raw if raw is not None else json.dumps(body), headers=headers
        )


def event(
    text: str = "web-01 có vấn đề gì?",
    *,
    sender: str = ALLOWED,
    chat: str | None = None,
    mid: str = "m-1",
    name: str = "message.text.received",
) -> dict:
    return {
        "event_name": name,
        "message": {
            "message_id": mid,
            "from": {"id": sender, "display_name": "An"},
            "chat": {"id": chat or sender, "chat_type": "PRIVATE"},
            "text": text,
        },
    }


def texts(sent: list[dict]) -> list[str]:
    return [m["body"]["text"] for m in sent]


# --------------------------------------------------------------------------- pure helpers
def test_parse_webhook_reads_top_level_and_nested_events():
    expected = {
        "message_id": "m-1",
        "sender_id": ALLOWED,
        "chat_id": ALLOWED,
        "display_name": "An",
        "text": "web-01 có vấn đề gì?",
    }
    assert zalo.parse_webhook(event()) == expected
    assert zalo.parse_webhook({"ok": True, "result": event()}) == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"event_name": "message.image.received", "message": {"from": {"id": "u"}, "text": "x"}},
        {"event_name": "message.text.received", "message": {"from": {"id": "u"}, "text": "  "}},
        {"event_name": "message.text.received", "message": {"text": "nobody sent this"}},
        {"event_name": "message.text.received"},
        {},
    ],
    ids=["image", "blank", "no-sender", "no-message", "empty"],
)
def test_parse_webhook_ignores_what_is_not_a_text_message(payload):
    assert zalo.parse_webhook(payload) is None


def test_split_message_respects_the_limit_and_keeps_order():
    para = "Đoạn văn. " * 150
    text = "\n\n".join([para, para, para])
    parts = zalo.split_message(text)
    assert len(parts) >= 3 and all(0 < len(p) <= zalo.TEXT_LIMIT for p in parts)
    squash = lambda s: s.replace("\n", "").replace(" ", "")  # noqa: E731
    assert squash("".join(parts)) == squash(text)
    assert [len(p) for p in zalo.split_message("x" * 4500)] == [2000, 2000, 500]
    assert zalo.split_message("ngắn") == ["ngắn"] and zalo.split_message("  ") == []


def test_seen_cache_expires_and_is_bounded():
    now = [0.0]
    cache = zalo.SeenCache(ttl=10, max_size=2, clock=lambda: now[0])
    assert cache.check_and_add("a") is False
    assert cache.check_and_add("a") is True
    now[0] = 11
    assert cache.check_and_add("a") is False  # expired
    cache.check_and_add("b")
    cache.check_and_add("c")  # "a" is dropped: max_size 2
    assert cache.check_and_add("a") is False


async def test_secret_cache_caches_refetches_and_never_caches_failures():
    now = [0.0]
    providers = FakeProviders({"p": "v1"})
    cache = zalo.SecretCache(10, fetch=providers, clock=lambda: now[0])
    assert await cache.get("p") == "v1"
    assert await cache.get("p") == "v1"
    assert providers.calls == ["p"]  # second read came from the cache
    now[0] = 11
    providers.values["p"] = "v2"
    assert await cache.get("p") == "v2"  # expired => refetched
    assert await cache.get("missing") is None
    assert await cache.get("missing") is None
    assert providers.calls.count("missing") == 2  # a failure is never cached
    providers.values["blank"] = "   "
    assert await cache.get("blank") is None


def test_secrets_default_to_access_control():
    assert zalo.SecretCache(1)._fetch is zalo.fetch_provider_secret


def test_ids_pass_the_agent_validators_and_sessions_rotate_daily():
    for raw in [ALLOWED, "A1b2C3", "id.with.dots", "weird id/../x", "é" * 5, "x" * 300]:
        validate_user_id(zalo.user_id_for(raw))
        validate_session_id(zalo.session_id_for(raw, NOW))
    assert zalo.session_id_for("c1", NOW) == "zalo-c1-20261007"
    next_day = datetime(2026, 10, 8, 0, 5, tzinfo=NOW.tzinfo)
    assert zalo.session_id_for("c1", NOW) != zalo.session_id_for("c1", next_day)


# --------------------------------------------------------------------------- webhook security
async def test_webhook_rejects_missing_or_wrong_secret(monkeypatch):
    agent = FakeAgent()
    ctx, app, sent = build(make_settings(monkeypatch), agent)
    assert (await post(app, event(), secret=None)).status_code == 403
    assert (await post(app, event(), secret="wrong")).status_code == 403
    await ctx.idle()
    assert agent.calls == [] and sent == []


async def test_webhook_fails_closed_when_the_secret_cannot_be_read(monkeypatch):
    agent = FakeAgent()
    ctx, app, _ = build(make_settings(monkeypatch), agent, providers=FakeProviders({}))
    assert (await post(app, event())).status_code == 503
    await ctx.idle()
    assert agent.calls == []


async def test_webhook_rejects_oversized_and_invalid_bodies(monkeypatch):
    _, app, _ = build(make_settings(monkeypatch), FakeAgent())
    assert (await post(app, None, raw="x" * (zalo.MAX_BODY_BYTES + 1))).status_code == 413
    assert (await post(app, None, raw="{not json")).status_code == 400


async def test_non_text_events_are_acknowledged_and_ignored(monkeypatch):
    agent = FakeAgent()
    ctx, app, sent = build(make_settings(monkeypatch), agent)
    r = await post(app, event(name="message.image.received"))
    assert r.status_code == 200 and r.json()["ignored"] is True
    await ctx.idle()
    assert agent.calls == [] and sent == []


async def test_a_retried_message_is_answered_once(monkeypatch):
    agent = FakeAgent("câu trả lời")
    ctx, app, sent = build(make_settings(monkeypatch), agent)
    assert (await post(app, event(mid="dup"))).json() == {"ok": True}
    assert (await post(app, event(mid="dup"))).json()["duplicate"] is True
    await ctx.idle()
    assert len(agent.calls) == 1 and texts(sent) == ["câu trả lời"]


# --------------------------------------------------------------------------- who may chat
async def test_allowed_user_gets_the_agent_answer(monkeypatch):
    agent = FakeAgent("web-01 quá tải CPU lúc 08:55 UTC.")
    ctx, app, sent = build(make_settings(monkeypatch), agent)
    r = await post(app, event("web-01 chậm lúc 15:40?", chat="chat-9"))
    assert r.status_code == 200
    await ctx.idle()
    assert agent.calls == [
        {
            "message": "web-01 chậm lúc 15:40?",
            "user_id": f"zalo-{ALLOWED}",
            "session_id": "zalo-chat-9-20261007",
        }
    ]
    (msg,) = sent
    assert msg["url"] == f"https://bot-api.zaloplatforms.com/bot{TOKEN}/sendMessage"
    assert msg["body"] == {"chat_id": "chat-9", "text": "web-01 quá tải CPU lúc 08:55 UTC."}


async def test_a_stranger_is_told_their_id_once_and_the_agent_is_not_called(monkeypatch):
    agent = FakeAgent()
    ctx, app, sent = build(make_settings(monkeypatch), agent)
    for i in range(3):
        r = await post(app, event("hi", sender="stranger-7", mid=f"s{i}"))
        assert r.json()["allowed"] is False
    await ctx.idle()
    assert agent.calls == []
    assert len(sent) == 1 and "stranger-7" in sent[0]["body"]["text"]


async def test_an_empty_allow_list_admits_nobody(monkeypatch):
    agent = FakeAgent()
    s = make_settings(monkeypatch, ZALO_ALLOWED_USER_IDS="[]")
    ctx, app, sent = build(s, agent)
    await post(app, event())
    await ctx.idle()
    assert agent.calls == [] and ALLOWED in texts(sent)[0]


# --------------------------------------------------------------------------- ordering and limits
async def test_each_chat_is_answered_in_order_while_chats_overlap(monkeypatch):
    s = make_settings(
        monkeypatch, ZALO_ALLOWED_USER_IDS=json.dumps(["a", "b"]), ZALO_MAX_WORKERS="2"
    )
    log_: list[tuple[str, str]] = []

    async def agent(message, *, user_id, session_id):
        log_.append(("start", message))
        await asyncio.sleep(0.05)
        log_.append(("end", message))
        return {"status": "success", "response": f"re:{message}"}

    ctx, app, sent = build(s, agent)
    await post(app, event("a1", sender="a", mid="1"))
    await post(app, event("a2", sender="a", mid="2"))
    await post(app, event("b1", sender="b", mid="3"))
    await ctx.idle()
    assert log_.index(("end", "a1")) < log_.index(("start", "a2"))  # same chat: strictly in order
    assert log_.index(("start", "b1")) < log_.index(("end", "a1"))  # other chat: not blocked
    assert [m["body"]["text"] for m in sent if m["body"]["chat_id"] == "a"] == ["re:a1", "re:a2"]


async def test_a_chat_that_floods_is_asked_to_slow_down(monkeypatch):
    s = make_settings(monkeypatch, ZALO_MAX_QUEUE_PER_CHAT="1")
    agent = FakeAgent()
    agent.gate = asyncio.Event()
    ctx, app, sent = build(s, agent)
    await post(app, event("q0", mid="m0"))
    await asyncio.sleep(0.02)  # q0 is now being answered (blocked on the gate)
    await post(app, event("q1", mid="m1"))  # waits behind q0
    await post(app, event("q2", mid="m2"))  # queue is full
    await asyncio.sleep(0.02)
    assert [c["message"] for c in agent.calls] == ["q0"]
    assert zalo.MSG_SLOW_DOWN in texts(sent)
    agent.gate.set()
    await ctx.idle()
    assert [c["message"] for c in agent.calls] == ["q0", "q1"]


async def test_the_still_working_notice_is_sent_only_when_the_answer_is_slow(monkeypatch):
    s = make_settings(monkeypatch, ZALO_NOTICE_AFTER_S="0.05")
    ctx, app, sent = build(s, FakeAgent("done", delay=0.25))
    await post(app, event(mid="slow"))
    await ctx.idle()
    assert texts(sent) == [zalo.MSG_WORKING, "done"]
    ctx2, app2, sent2 = build(s, FakeAgent("quick"))
    await post(app2, event(mid="fast"))
    await ctx2.idle()
    assert texts(sent2) == ["quick"]


# --------------------------------------------------------------------------- failures
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (503, zalo.MSG_BUSY),
        (429, zalo.MSG_BUSY),
        (502, zalo.MSG_BUSY),
        (504, zalo.MSG_TIMEOUT),
        (422, zalo.MSG_TOO_MANY_STEPS),
        (500, zalo.MSG_ERROR),
        (400, zalo.MSG_ERROR),
    ],
)
async def test_agent_failures_become_friendly_replies(monkeypatch, status, expected):
    agent = FakeAgent(error=GreenNodeRequestError("internal detail: key=abc", status_code=status))
    ctx, app, sent = build(make_settings(monkeypatch), agent)
    await post(app, event())
    await ctx.idle()
    assert texts(sent) == [expected]


async def test_unexpected_errors_and_odd_results_never_leak_details(monkeypatch):
    ctx, app, sent = build(
        make_settings(monkeypatch), FakeAgent(error=RuntimeError("boom: key=abc"))
    )
    await post(app, event())
    await ctx.idle()
    assert texts(sent) == [zalo.MSG_ERROR]

    async def interrupted(message, *, user_id, session_id):
        return {"status": "interrupted", "interrupt": {"tool_calls": []}}

    async def empty(message, *, user_id, session_id):
        return {"status": "success", "response": "  "}

    for agent, expected in ((interrupted, zalo.MSG_ERROR), (empty, zalo.MSG_EMPTY)):
        ctx, app, sent = build(make_settings(monkeypatch), agent)
        await post(app, event())
        await ctx.idle()
        assert texts(sent) == [expected]


async def test_long_answers_are_split_into_several_messages(monkeypatch):
    ctx, app, sent = build(make_settings(monkeypatch), FakeAgent("x" * 4500))
    await post(app, event())
    await ctx.idle()
    assert [len(t) for t in texts(sent)] == [2000, 2000, 500]


async def test_the_bot_token_never_reaches_the_logs(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)

    def exploding(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot connect to {request.url}")  # the URL carries the token

    ctx, app, _ = build(make_settings(monkeypatch), FakeAgent("ok"), zalo_handler=exploding)
    await post(app, event())
    await ctx.idle()
    assert TOKEN not in caplog.text and SECRET not in caplog.text
    assert "reply not delivered" in caplog.text and "ConnectError" in caplog.text


# --------------------------------------------------------------------------- settings
@pytest.mark.parametrize(
    ("env", "needle"),
    [
        ({"ZALO_API_BASE": "http://insecure.example"}, "https"),
        ({"ZALO_TOKEN_PROVIDER": "bad name!"}, "provider"),
        ({"ZALO_TIMEZONE": "Mars/Base"}, "ZALO_TIMEZONE"),
    ],
    ids=["http-base", "provider-name", "timezone"],
)
def test_enabling_zalo_validates_its_settings(monkeypatch, env, needle):
    monkeypatch.setenv("ZALO_ENABLED", "true")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    with pytest.raises(ValueError, match=needle):
        Settings()


# --------------------------------------------------------------------------- plain text
def test_markdown_is_stripped_but_identifiers_and_bullets_survive():
    src = "## Kết luận\n**web-01** quá tải: `cpu_usage_pct` ~90%\n- es_get_metrics và web_01 giữ nguyên\n**dở dang"
    assert zalo.to_plain_text(src) == (
        "Kết luận\nweb-01 quá tải: cpu_usage_pct ~90%\n- es_get_metrics và web_01 giữ nguyên\n**dở dang"
    )


async def test_the_answer_is_sent_as_plain_text(monkeypatch):
    ctx, app, sent = build(make_settings(monkeypatch), FakeAgent("**web-01** chậm do `CPU`"))
    await post(app, event())
    await ctx.idle()
    assert texts(sent) == ["web-01 chậm do CPU"]


# --------------------------------------------------------------------------- log hygiene
async def test_httpx_request_logs_do_not_contain_the_bot_token(monkeypatch, caplog):
    """httpx logs every request URL at INFO and the Zalo URL carries the bot token (seen in the runtime log)."""
    caplog.set_level(logging.INFO)
    caplog.set_level(
        logging.INFO, logger="httpx"
    )  # langfuse, imported by other tests, raises it to WARNING
    ctx, app, sent = build(make_settings(monkeypatch), FakeAgent("ok"))
    await post(app, event())
    await ctx.idle()
    assert sent, "a reply must have gone out through the fake Zalo API"
    assert TOKEN not in caplog.text
    assert "https://bot-api.zaloplatforms.com/bot<redacted>/sendMessage" in caplog.text


def test_install_log_redaction_is_idempotent():
    zalo.install_log_redaction()
    zalo.install_log_redaction()
    filters = [f for f in logging.getLogger("httpx").filters if isinstance(f, zalo.RedactBotToken)]
    assert len(filters) == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "POST https://bot-api.zaloplatforms.com/bot{t}/sendMessage",
            "POST https://bot-api.zaloplatforms.com/bot<redacted>/sendMessage",
        ),
        (
            "GET https://bot-api.zaloplatforms.com:443/bot{t}/getMe",
            "GET https://bot-api.zaloplatforms.com:443/bot<redacted>/getMe",
        ),
        (
            "for url 'https://bot-api.zaloplatforms.com/bot{t}'",
            "for url 'https://bot-api.zaloplatforms.com/bot<redacted>'",
        ),
        (
            "nothing secret: https://bot-api.zaloplatforms.com/health",
            "nothing secret: https://bot-api.zaloplatforms.com/health",
        ),
    ],
    ids=["send", "port", "quoted-without-method", "untouched"],
)
def test_redaction_masks_only_the_token_not_the_host(raw, expected):
    record = logging.LogRecord("httpx", logging.INFO, __file__, 1, raw.format(t=TOKEN), None, None)
    zalo.RedactBotToken().filter(record)
    assert record.getMessage() == expected


def test_the_redaction_filter_handles_non_string_arguments():
    url = httpx.URL(f"https://bot-api.zaloplatforms.com/bot{TOKEN}/sendMessage")
    record = logging.LogRecord(
        "httpx", logging.INFO, __file__, 1, "HTTP Request: %s %s", ("POST", url), None
    )
    assert zalo.RedactBotToken().filter(record) is True
    assert record.getMessage() == (
        "HTTP Request: POST https://bot-api.zaloplatforms.com/bot<redacted>/sendMessage"
    )
    plain = logging.LogRecord("x", logging.INFO, __file__, 1, "nothing secret %s", ("here",), None)
    zalo.RedactBotToken().filter(plain)
    assert plain.getMessage() == "nothing secret here"
