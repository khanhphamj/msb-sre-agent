"""Zalo Bot channel: the bot's webhook comes in here, replies go out through the Zalo Bot API.

Flow (same container :8080, next to /invocations):
  Zalo ─POST /webhook/zalo (X-Bot-Api-Secret-Token)─▶ verify ▶ ack 200 at once
       └▶ per-chat queue ▶ service.run_chat(...) ▶ sendMessage (several messages when > 2000 chars)

Security
  - The bot token and the webhook secret live in AgentBase Access Control as Static API Key providers and are read
    through app/identity.py (the runtime's agent identity) — never from .env. A failed read fails closed.
  - The webhook is verified in constant time BEFORE the body is read: no secret available => 503, wrong => 403.
  - Only Zalo ids listed in ZALO_ALLOWED_USER_IDS are answered; anyone else is told their own id (once per 10 min).
  - The bot token is part of every Zalo API URL: never log a URL or an httpx exception text, only its type.

State (dedupe cache, queues, secret cache) is per process: run ONE replica, or accept that a message Zalo retries
may be answered twice after a restart.

Zalo Bot API: POST https://bot-api.zaloplatforms.com/bot<TOKEN>/<method> — sendMessage {chat_id, text} (≤ 2000
characters), setWebhook {url, secret_token}; the webhook body is {"event_name": "message.text.received",
"message": {"from": {"id"}, "chat": {"id"}, "text", "message_id"}} (docs may nest it in "result").
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import secrets
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from greennode_agentbase.exceptions import GreenNodeRequestError
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import BaseRoute, Route

from app.config import Settings, get_settings
from app.identity import agent_api_key

log = logging.getLogger(__name__)

WEBHOOK_PATH = "/webhook/zalo"
TEXT_LIMIT = 2000  # Zalo sendMessage text limit
MAX_BODY_BYTES = 64 * 1024
SEND_TIMEOUT_S = 20.0
DEDUPE_TTL_S = 600.0
DEDUPE_MAX = 10_000
DENIED_TTL_S = 600.0  # a stranger gets the "not authorised" reply at most once per this time

MSG_WORKING = "Mình đang tra cứu dữ liệu, bạn chờ chút nhé…"
MSG_BUSY = "Hệ thống đang bận, bạn vui lòng thử lại sau ít phút nhé."
MSG_TIMEOUT = "Câu hỏi này mất quá nhiều thời gian để xử lý. Bạn thử hỏi cụ thể hơn (host, khoảng thời gian) nhé."
MSG_TOO_MANY_STEPS = "Câu hỏi cần quá nhiều bước tra cứu. Bạn hãy chia nhỏ câu hỏi hoặc nêu rõ host và khoảng thời gian nhé."
MSG_ERROR = "Xin lỗi, mình chưa xử lý được yêu cầu này. Bạn thử lại sau nhé."
MSG_EMPTY = "Mình chưa có câu trả lời cho yêu cầu này. Bạn thử diễn đạt lại giúp mình nhé."
MSG_SLOW_DOWN = (
    "Bạn gửi nhanh quá, mình đang xử lý các tin trước. Bạn chờ một chút rồi gửi lại nhé."
)
MSG_DENIED = (
    "Bạn chưa được cấp quyền dùng bot này. Zalo ID của bạn là: {sender_id}\n"
    "Hãy gửi ID này cho quản trị viên để được thêm vào danh sách."
)


def reply_for_status(status: int | None) -> str:
    """What a user sees when the agent request fails (never a technical detail)."""
    if status == 504:
        return MSG_TIMEOUT
    if status == 422:
        return MSG_TOO_MANY_STEPS
    if status in (429, 502, 503):
        return MSG_BUSY
    return MSG_ERROR


_MARKDOWN = (
    (re.compile(r"\*\*(.+?)\*\*", re.S), r"\1"),  # **bold**
    (re.compile(r"`([^`\n]+)`"), r"\1"),  # `code`
    (re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*", re.M), ""),  # # headings
)


def to_plain_text(text: str) -> str:
    """Zalo shows our text as is (no parse_mode): drop the markdown that LLMs add anyway. Identifiers such as
    web_01 or es_get_metrics and "- " bullets are left alone."""
    for pattern, repl in _MARKDOWN:
        text = pattern.sub(repl, text)
    return text


# --------------------------------------------------------------------------- ids
_SAFE_ID = re.compile(r"[A-Za-z0-9-]{1,100}")


def _safe(raw: str) -> str:
    """Zalo ids are opaque: keep them readable when they only use [A-Za-z0-9-], otherwise hash them, so the
    result always passes app.auth.inbound.validate_user_id / validate_session_id."""
    return raw if _SAFE_ID.fullmatch(raw) else hashlib.sha256(raw.encode()).hexdigest()[:32]


def user_id_for(sender_id: str) -> str:
    return f"zalo-{_safe(sender_id)}"


def session_id_for(chat_id: str, now: datetime) -> str:
    """One session per chat per local day: the context stays bounded and an investigation starts fresh."""
    return f"zalo-{_safe(chat_id)}-{now:%Y%m%d}"


# --------------------------------------------------------------------------- webhook payload
def _event_body(payload: dict) -> dict:
    # Zalo really sends event_name/message at the TOP level; the docs show them nested in "result".
    if "event_name" in payload:
        return payload
    return payload.get("result") or payload.get("data") or {}


def parse_webhook(payload: dict) -> dict | None:
    """message_id, sender_id, chat_id, display_name, text of a text message from an identifiable sender, else None
    (image, sticker, voice, empty text, other events)."""
    body = _event_body(payload)
    if "message.text" not in str(body.get("event_name") or body.get("eventName") or ""):
        return None
    msg = body.get("message") or {}
    text = str(msg.get("text") or "").strip()
    if not text:
        return None
    sender = msg.get("from") or {}
    chat = msg.get("chat") or {}
    sender_id = str(sender.get("id") or chat.get("id") or "")
    chat_id = str(chat.get("id") or sender.get("id") or "")
    if not sender_id or not chat_id:
        return None
    return {
        "message_id": str(msg.get("message_id") or msg.get("messageId") or ""),
        "sender_id": sender_id,
        "chat_id": chat_id,
        "display_name": str(sender.get("display_name") or sender.get("displayName") or ""),
        "text": text,
    }


def split_message(text: str, limit: int = TEXT_LIMIT) -> list[str]:
    """Parts of at most `limit` characters, in order, cut at a paragraph/line/sentence/space in the second half of the
    window when there is one. No part is empty."""
    text = str(text or "").strip()
    parts: list[str] = []
    while len(text) > limit:
        cut = limit
        for sep, keep in (("\n\n", 0), ("\n", 0), (". ", 1), ("! ", 1), ("? ", 1), (" ", 0)):
            idx = text.rfind(sep, 0, limit)
            if idx > limit // 2:
                cut = idx + keep
                break
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts


class SeenCache:
    """Bounded in-memory set of recently seen keys (entries expire after `ttl`, oldest dropped beyond `max_size`)."""

    def __init__(
        self,
        ttl: float = DEDUPE_TTL_S,
        max_size: int = DEDUPE_MAX,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._ttl, self._max_size, self._clock = ttl, max_size, clock
        self._seen: OrderedDict[str, float] = OrderedDict()

    def check_and_add(self, key: str) -> bool:
        """Record `key`; True when it was already seen (and has not expired)."""
        now = self._clock()
        while self._seen and next(iter(self._seen.values())) + self._ttl <= now:
            self._seen.popitem(last=False)
        if key in self._seen:
            return True
        self._seen[key] = now
        while len(self._seen) > self._max_size:
            self._seen.popitem(last=False)
        return False


# --------------------------------------------------------------------------- secrets (Access Control)
async def fetch_provider_secret(provider_name: str) -> str:
    """The value of one Static API Key provider (the runtime's agent identity and IAM come from the platform)."""

    @agent_api_key(provider_name)
    async def _read(*, api_key: str) -> str:
        return api_key

    return await _read()


class SecretCache:
    """Per-process cache of provider values: secrets rarely change and reading Identity on every webhook would add
    a platform call to each one. A failed or empty read is never cached and yields None (callers fail closed)."""

    def __init__(
        self,
        ttl_s: float,
        fetch: Callable[[str], Awaitable[str]] = fetch_provider_secret,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._ttl, self._fetch, self._clock = ttl_s, fetch, clock
        self._values: dict[str, tuple[float, str]] = {}

    async def get(self, provider: str) -> str | None:
        now = self._clock()
        hit = self._values.get(provider)
        if hit and now < hit[0]:
            return hit[1]
        try:
            value = (await self._fetch(provider)).strip()
        except Exception as e:  # noqa: BLE001 — Identity down, provider missing, no agent identity...
            log.warning(
                "zalo: cannot read provider %r from Access Control (%s)", provider, type(e).__name__
            )
            return None
        if not value:
            log.warning("zalo: provider %r is empty", provider)
            return None
        self._values[provider] = (now + self._ttl, value)
        return value


# --------------------------------------------------------------------------- Zalo Bot API client
# httpx logs every request URL at INFO, and a Zalo URL is /bot<token>/<method>: unfiltered, the token lands in the logs.
# The lookbehind skips the host: "//bot-api.zaloplatforms.com" also starts with "/bot".
_BOT_TOKEN_IN_URL = re.compile(r"""(?<!/)/bot[^/\s"'?#]+""")


class RedactBotToken(logging.Filter):
    """Masks the bot token in a record, whatever its args are (httpx passes an httpx.URL, not a string)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except (TypeError, ValueError):
            return True  # malformed record: leave it to logging's own error handling
        redacted = _BOT_TOKEN_IN_URL.sub("/bot<redacted>", message)
        if redacted != message:
            record.msg, record.args = redacted, ()
        return True


def install_log_redaction() -> None:
    """Idempotent. A logger's filters only see the records that logger creates, so it goes on the loggers that print URLs."""
    for name in ("httpx", "httpcore"):
        logger = logging.getLogger(name)
        if not any(isinstance(f, RedactBotToken) for f in logger.filters):
            logger.addFilter(RedactBotToken())


class ZaloClient:
    def __init__(
        self,
        settings: Settings,
        secrets_cache: SecretCache,
        transport: httpx.AsyncBaseTransport | None = None,  # tests inject httpx.MockTransport
    ):
        self._base = settings.zalo_api_base.rstrip("/")
        self._token_provider = settings.zalo_token_provider
        self._secrets = secrets_cache
        self._http = httpx.AsyncClient(timeout=SEND_TIMEOUT_S, transport=transport)
        install_log_redaction()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def call(self, method: str, body: dict | None = None, *, get: bool = False) -> dict:
        """One Zalo Bot API call. Returns Zalo's JSON, or {"ok": False, "error"|"http": ...}. Never raises and never
        logs the URL itself (it contains the bot token); httpx's own request log is masked by RedactBotToken."""
        token = await self._secrets.get(self._token_provider)
        if not token:
            return {"ok": False, "error": "bot token unavailable"}
        url = f"{self._base}/bot{token}/{method}"
        try:
            r = await (self._http.get(url) if get else self._http.post(url, json=body or {}))
            data = r.json() if r.status_code == 200 else {"ok": False, "http": r.status_code}
        except (httpx.HTTPError, ValueError) as e:
            return {"ok": False, "error": type(e).__name__}
        return data if isinstance(data, dict) else {"ok": False, "error": "unexpected response"}

    async def send_message(self, chat_id: str, text: str) -> dict:
        """Send `text` (several messages when it exceeds 2000 characters), in order, stopping at the first failure."""
        parts = split_message(text)
        if not parts:
            return {"ok": False, "sent": 0, "parts": 0, "error": "empty message"}
        for sent, part in enumerate(parts):
            answer = await self.call("sendMessage", {"chat_id": chat_id, "text": part})
            if not answer.get("ok"):
                why = (
                    answer.get("error")
                    or answer.get("http")
                    or answer.get("description")
                    or "rejected"
                )
                return {"ok": False, "sent": sent, "parts": len(parts), "error": str(why)[:200]}
        return {"ok": True, "parts": len(parts)}


# --------------------------------------------------------------------------- per-chat ordering
class ChatDispatcher:
    """Answer chats concurrently (bounded) but every chat's messages one at a time, in arrival order."""

    def __init__(
        self, handler: Callable[[dict], Awaitable[None]], *, max_workers: int, max_queue: int
    ):
        self._handler = handler
        self._slots = asyncio.Semaphore(max(1, max_workers))
        self._max_queue = max(1, max_queue)
        self._queues: dict[str, deque[dict]] = {}
        self._tasks: set[asyncio.Task] = set()

    def submit(self, chat_id: str, item: dict) -> bool:
        """Queue `item`; False when this chat already has `max_queue` messages waiting (caller says "slow down")."""
        queue = self._queues.get(chat_id)
        if queue is not None:  # a worker is draining this chat and will pick the item up
            if len(queue) >= self._max_queue:
                return False
            queue.append(item)
            return True
        self._queues[chat_id] = deque([item])
        task = asyncio.create_task(self._drain(chat_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def _drain(self, chat_id: str) -> None:
        queue = self._queues[chat_id]
        while queue:  # no await between this check and the `del` below, so no item can slip in
            item = queue.popleft()
            async with self._slots:
                try:
                    await self._handler(item)
                except Exception as e:  # noqa: BLE001 — one failing message must not stop the chat
                    log.error("zalo: handler failed (%s)", type(e).__name__)
        del self._queues[chat_id]

    async def idle(self) -> None:
        """Wait until every queued message has been handled (tests, shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)


# --------------------------------------------------------------------------- the channel
async def _service_run_chat(message: str, *, user_id: str, session_id: str) -> dict:
    from app import service  # lazy: app.service imports the whole agent

    return await service.run_chat(message, user_id=user_id, session_id=session_id)


class ZaloContext:
    """Everything the webhook needs; injectable pieces keep the tests offline."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: ZaloClient | None = None,
        secrets_cache: SecretCache | None = None,
        run_chat: Callable[..., Awaitable[dict]] | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self.settings = settings
        self.secrets = secrets_cache or SecretCache(settings.zalo_secret_ttl_s)
        self.client = client or ZaloClient(settings, self.secrets)
        self._run_chat = run_chat or _service_run_chat
        zone = ZoneInfo(settings.zalo_timezone)
        self._now = clock or (lambda: datetime.now(zone))
        self.seen = SeenCache()
        self.denied = SeenCache(ttl=DENIED_TTL_S)
        self._allowed = frozenset(i.strip() for i in settings.zalo_allowed_user_ids if i.strip())
        self.dispatcher = ChatDispatcher(
            self.process,
            max_workers=settings.zalo_max_workers,
            max_queue=settings.zalo_max_queue_per_chat,
        )
        self._background: set[asyncio.Task] = set()

    def is_allowed(self, sender_id: str) -> bool:
        return sender_id in self._allowed

    def notify(self, chat_id: str, text: str) -> None:
        """Fire-and-forget message that needs no LLM (not-authorised, slow down)."""
        self._spawn(self._send_quietly(chat_id, text))

    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _send_quietly(self, chat_id: str, text: str) -> None:
        result = await self.client.send_message(chat_id, text)
        if not result.get("ok"):
            log.warning("zalo: could not send a notice (%s)", result.get("error"))

    async def _notice_later(self, chat_id: str) -> None:
        try:
            await asyncio.sleep(self.settings.zalo_notice_after_s)
            await self._send_quietly(chat_id, MSG_WORKING)
        except asyncio.CancelledError:  # the answer arrived in time
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("zalo: notice failed (%s)", type(e).__name__)

    async def _answer(self, sender_id: str, chat_id: str, text: str) -> str:
        try:
            result = await self._run_chat(
                text,
                user_id=user_id_for(sender_id),
                session_id=session_id_for(chat_id, self._now()),
            )
        except GreenNodeRequestError as e:
            log.warning("zalo: agent request failed (status=%s)", e.status_code)
            return reply_for_status(e.status_code)
        except Exception as e:  # noqa: BLE001
            log.error("zalo: agent failed (%s)", type(e).__name__)
            return MSG_ERROR
        if result.get("status") != "success":  # "interrupted" cannot happen without HITL tools
            return MSG_ERROR
        return to_plain_text(str(result.get("response") or "")).strip() or MSG_EMPTY

    async def process(self, item: dict) -> None:
        """Answer one message (called by the dispatcher, one chat at a time)."""
        chat_id = item["chat_id"]
        notice = None
        if self.settings.zalo_notice_after_s > 0:
            notice = asyncio.ensure_future(self._notice_later(chat_id))
        try:
            answer = await self._answer(item["sender_id"], chat_id, item["text"])
        finally:
            if notice is not None:
                notice.cancel()
        sent = await self.client.send_message(chat_id, answer)
        if not sent.get("ok"):
            log.warning(
                "zalo: reply not delivered (sent %s/%s: %s)",
                sent.get("sent"),
                sent.get("parts"),
                sent.get("error"),
            )

    async def idle(self) -> None:
        """Wait for queued messages and fire-and-forget notices to finish (tests, shutdown)."""
        await self.dispatcher.idle()
        while self._background:
            await asyncio.gather(*list(self._background), return_exceptions=True)

    async def aclose(self) -> None:
        await self.client.aclose()


_ctx: ZaloContext | None = None


def zalo_routes(
    settings: Settings | None = None, *, ctx: ZaloContext | None = None
) -> list[BaseRoute]:
    """The webhook route. `ctx` is injected by tests; the app builds it from settings."""
    global _ctx
    settings = settings or get_settings()
    ctx = ctx or ZaloContext(settings)
    _ctx = ctx
    if not settings.zalo_allowed_user_ids:
        log.warning(
            "ZALO_ALLOWED_USER_IDS is empty: every sender will only be told their own Zalo id."
        )

    async def webhook(request: Request) -> JSONResponse:
        expected = await ctx.secrets.get(settings.zalo_secret_provider)
        if not expected:  # fail closed: without the secret nobody can be verified
            return JSONResponse({"error": "webhook secret unavailable"}, status_code=503)
        given = request.headers.get("x-bot-api-secret-token", "")
        if not secrets.compare_digest(given.encode(), expected.encode()):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return JSONResponse({"error": "payload too large"}, status_code=413)
        try:
            payload: Any = json.loads(body)
        except ValueError:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        update = parse_webhook(payload) if isinstance(payload, dict) else None
        if update is None:  # image, sticker, voice, other events: acknowledge, do nothing
            return JSONResponse({"ok": True, "ignored": True})
        message_id = update["message_id"]
        if message_id and ctx.seen.check_and_add(
            message_id
        ):  # Zalo retried a message we already have
            return JSONResponse({"ok": True, "duplicate": True})
        sender_id, chat_id = update["sender_id"], update["chat_id"]
        if not ctx.is_allowed(sender_id):
            log.info("zalo: sender %s is not in ZALO_ALLOWED_USER_IDS", sender_id)
            if not ctx.denied.check_and_add(sender_id):
                ctx.notify(chat_id, MSG_DENIED.format(sender_id=sender_id))
            return JSONResponse({"ok": True, "allowed": False})
        if not ctx.dispatcher.submit(chat_id, update):
            ctx.notify(chat_id, MSG_SLOW_DOWN)
        return JSONResponse({"ok": True})

    return [Route(WEBHOOK_PATH, webhook, methods=["POST"])]


async def shutdown_zalo() -> None:
    if _ctx is not None:
        await _ctx.aclose()
