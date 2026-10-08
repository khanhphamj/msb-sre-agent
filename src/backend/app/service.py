"""Orchestrates one request: auth -> tools -> graph -> invoke/stream, wrapped in a Langfuse trace.

Payload contract (POST /invocations) — keep stable, the frontend & eval depend on it:

  {"type": "chat",     "message": "...", "stream": false}
  {"type": "resume",   "decisions": [{"tool_call_id","action":"approve|edit|reject","args"?,"reason"?}],
                       "stream": false}                                    # HITL
  {"type": "feedback", "trace_id": "...", "feedback_token": "...", "score": 1, "comment": "..."}
                       # score: 1 | 0 | -1; feedback_token comes from the response — binds the trace to the user

Headers:
  Authorization: Bearer <user JWT>                 (or AUTH_TOKEN_HEADER)
  X-GreenNode-AgentBase-Session-Id: <uuid>         (required for chat/resume)
  X-GreenNode-AgentBase-User-Id: <sub>             (optional; if present must match the token)

Response non-stream:
  {"status": "success",     "response", "tools_used", "session_id", "trace_id", "feedback_token"}
  {"status": "interrupted", "interrupt": {"id","type","tool_calls","message"}, "session_id", "trace_id"}
Response stream (SSE, one `data: {...}` per line):
  token | tool_start | tool_end | reset (self-eval forces a re-answer) |
  interrupt {interrupt} | done {response, tools_used, trace_id} | error {message}
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import logging
import time
import weakref
from collections.abc import AsyncIterator
from typing import Any

import openai
from greennode_agentbase import GreenNodeAgentBaseContext, RequestContext
from greennode_agentbase.exceptions import GreenNodeRequestError
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from app.auth.inbound import (
    Principal,
    authenticate_async,
    validate_session_id,
    validate_user_id,
)
from app.config import get_settings
from app.graph.builder import build_graph, recursion_limit
from app.hitl import (
    interrupt_payload,
    is_placeholder_tool_message,
    is_waiting_approval,
    last_ai,
    requires_approval,
    validate_decisions,
)
from app.llm import TIERS, fallbacks_for, model_for
from app.memory.short_term import get_checkpointer, thread_config
from app.observability import tracing
from app.tools import collect_tools
from app.tools.mcp import reset_policy_denials

log = logging.getLogger(__name__)
MAX_MESSAGE_CHARS = 16_000


def _error(msg: str, status: int = 400) -> GreenNodeRequestError:
    return GreenNodeRequestError(msg, status_code=status)


async def handle(payload: dict[str, Any], context: RequestContext) -> Any:
    kind = payload.get("type", "chat")
    principal = await authenticate_async(context, get_settings())

    if kind == "feedback":
        return _handle_feedback(payload, principal)
    if kind not in ("chat", "resume"):
        raise _error(f"Unknown payload type: {kind}")
    if not context.session_id:
        raise _error("Header X-GreenNode-AgentBase-Session-Id is required")
    validate_session_id(context.session_id)  # prevents path traversal into another user's session

    if kind == "chat":
        message = (payload.get("message") or "").strip()
        if not message:
            raise _error("`message` is required")
        if len(message) > MAX_MESSAGE_CHARS:
            raise _error(f"`message` exceeds {MAX_MESSAGE_CHARS} chars")
        graph_input: Any = {"messages": [HumanMessage(message)]}
        trace_input: Any = message
    else:
        try:
            decisions = validate_decisions(payload.get("decisions"))
        except ValueError as e:
            raise _error(str(e)) from e
        graph_input = Command(resume=decisions)
        trace_input = {"resume": decisions, "interrupt_id": payload.get("interrupt_id")}

    run = _Run(
        principal,
        context.session_id,
        kind,
        graph_input,
        trace_input,
        interrupt_id=payload.get("interrupt_id"),
    )
    if payload.get("stream"):
        return run.stream()
    return await run.invoke()


async def run_chat(message: str, *, user_id: str, session_id: str) -> dict:
    """Call the agent in-process, BYPASSING inbound auth — only for eval / tests / internal jobs / A2A."""
    validate_user_id(user_id)
    validate_session_id(session_id)
    message = (message or "").strip()
    if not message:
        raise _error("`message` is required")
    if len(message) > MAX_MESSAGE_CHARS:
        raise _error(f"`message` exceeds {MAX_MESSAGE_CHARS} chars")
    run = _Run(
        Principal(user_id=user_id),
        session_id,
        "chat",
        {"messages": [HumanMessage(message)]},
        message,
    )
    return await run.invoke()


async def run_resume(decisions: list[dict], *, user_id: str, session_id: str) -> dict:
    """Resume HITL in-process — only for eval / tests / A2A."""
    validate_user_id(user_id)
    validate_session_id(session_id)
    decisions = validate_decisions(decisions)
    run = _Run(
        Principal(user_id=user_id),
        session_id,
        "resume",
        Command(resume=decisions),
        {"resume": decisions},
    )
    return await run.invoke()


def feedback_token(user_id: str, trace_id: str) -> str:
    """HMAC(user_id, trace_id): only the user who received the answer can score that trace."""
    msg = f"{user_id}\x00{trace_id}".encode()
    return hmac.new(get_settings().signing_key, msg, hashlib.sha256).hexdigest()


async def pending_tool_calls(*, user_id: str, session_id: str) -> list[dict]:
    """Tool calls pending approval (HITL) for exactly this (user, session) — used by A2A."""
    s = get_settings()
    graph = build_graph(settings=s, tools=[], checkpointer=get_checkpointer(s))
    snap = await graph.aget_state(thread_config(s, session_id=session_id, user_id=user_id))
    if not is_waiting_approval(snap):
        return []
    ai = last_ai(snap.values.get("messages", []))
    return [
        {"id": tc["id"], "name": tc["name"], "args": tc["args"]}
        for tc in (ai.tool_calls if ai else [])
        if requires_approval(tc["name"], s)
    ]


def _handle_feedback(payload: dict[str, Any], principal: Principal) -> dict:
    trace_id = payload.get("trace_id")
    score = payload.get("score")
    if not trace_id or score not in (-1, 0, 1):
        raise _error("feedback requires trace_id and score in {-1, 0, 1}")
    token = str(payload.get("feedback_token") or "")
    if not hmac.compare_digest(token, feedback_token(principal.user_id, trace_id)):
        raise _error("feedback_token is not valid for this user", 403)
    ok = tracing.record_feedback(
        trace_id=trace_id, value=float(score), comment=payload.get("comment")
    )
    return {"status": "success" if ok else "ignored"}


def _trace_meta(kind: str) -> dict[str, Any]:
    s = get_settings()
    return {
        "request_id": GreenNodeAgentBaseContext.get_request_id(),
        "request_type": kind,
        **{f"llm_model_{t}": model_for(t) for t in TIERS},
        **{f"llm_fallbacks_{t}": ",".join(fallbacks_for(t)) for t in TIERS},
        "llm_adaptive_routing": s.llm_adaptive_routing,
        "memory_backend": s.memory_backend,
        "memory_id": s.memory_id,
        "auth_mode": s.auth_mode,
        "hitl_tools": ",".join(s.hitl_tools),
        "reflection_enabled": s.reflection_enabled,
    }


def _trace_auth(principal: Principal) -> None:
    claims = principal.claims
    tracing.event(
        "auth",
        output={
            "user_id": principal.user_id,
            "mode": get_settings().auth_mode,
            "iss": claims.get("iss"),
            "aud": claims.get("aud"),
            "exp": claims.get("exp"),
            "client": claims.get("azp") or claims.get("client_id"),
        },
    )


def _final_text(messages: list) -> str:
    last = messages[-1] if messages else None
    if isinstance(last, AIMessage) and not last.tool_calls:
        return last.text
    return ""


def _tools_used(messages: list) -> list[str]:
    """Names of tools run in the current turn (after the last HumanMessage)."""
    used: list[str] = []
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, ToolMessage) and m.name and not is_placeholder_tool_message(m):
            used.append(m.name)
    return list(reversed(used))


# A (user, session) runs only 1 request at a time within the process: blocks double-tap resume
# (approved tool running twice) and 2 parallel chats overwriting the checkpoint. Across replicas: see interrupt_id.
_SESSION_LOCKS: weakref.WeakValueDictionary[tuple[str, str], asyncio.Lock] = (
    weakref.WeakValueDictionary()
)


def _session_lock(user_id: str, session_id: str) -> asyncio.Lock:
    key = (user_id, session_id)
    lock = _SESSION_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _SESSION_LOCKS[key] = lock
    return lock


def _map_error(e: BaseException) -> GreenNodeRequestError:
    """Infrastructure errors ⇒ meaningful HTTP codes instead of a generic 500."""
    if isinstance(e, GreenNodeRequestError):
        status = getattr(e, "status_code", 500) or 500
        if status in (429,) or status >= 500:  # Memory/Identity overloaded or failing
            return _error("Backing service temporarily overloaded, try again later", 503)
        return e
    if isinstance(e, GraphRecursionError):
        return _error(
            "The request needs too many steps — please split up the question (MAX_TOOL_ROUNDS)", 422
        )
    if isinstance(e, openai.APIError):  # after all fallbacks have been tried
        return _error("LLM provider error or overloaded (fallbacks tried)", 502)
    if isinstance(e, TimeoutError):
        return _error(f"Agent timed out ({get_settings().request_timeout_s:.0f}s)", 504)
    return _error("Agent failed to process the request", 500)


class _Run:
    def __init__(
        self,
        principal: Principal,
        session_id: str,
        kind: str,
        graph_input,
        trace_input,
        interrupt_id: str | None = None,
    ):
        self.interrupt_id = interrupt_id
        self.settings = get_settings()
        self.principal = principal
        self.session_id = session_id
        self.kind = kind
        self.graph_input = graph_input
        self.trace_input = trace_input

    def _trace(self, tags: list[str]):
        return tracing.trace_request(
            settings=self.settings,
            user_id=self.principal.user_id,
            session_id=self.session_id,
            input=self.trace_input,
            metadata=_trace_meta(self.kind),
            tags=[f"type:{self.kind}", *tags],
        )

    async def _prepare(self):
        reset_policy_denials()
        # Identity SDK keys 3LO / delegated credentials by this context user id — set it from the VERIFIED
        # principal on every path (HTTP, A2A, run_chat for eval/jobs), never leave it unset or header-derived.
        GreenNodeAgentBaseContext.set_user_id(self.principal.user_id)
        _trace_auth(self.principal)
        tools = await collect_tools(self.settings, self.principal)
        graph = build_graph(
            settings=self.settings, tools=tools, checkpointer=get_checkpointer(self.settings)
        )
        claims = self.principal.claims or {}
        config = thread_config(
            self.settings,
            session_id=self.session_id,
            user_id=self.principal.user_id,
            # only allowlisted claims — tools read them via config["configurable"]["user_claims"]
            user_claims={k: claims[k] for k in self.settings.auth_forward_claims if k in claims},
        )
        config["callbacks"] = tracing.langchain_callbacks()
        config["recursion_limit"] = recursion_limit(self.settings)
        config["run_name"] = "agent.graph"

        with tracing.step(
            "memory.checkpoint_load", metadata={"backend": self.settings.memory_backend}
        ):
            snapshot = await graph.aget_state(config)
        pending = is_waiting_approval(snapshot)
        if self.kind == "chat" and pending:
            raise _error("Awaiting tool approval — send {type: resume} before chatting again", 409)
        if self.kind == "resume" and not pending:
            raise _error("No approval request is pending in this session", 409)
        if self.kind == "resume" and self.interrupt_id:
            visible = [i.id for i in (getattr(snapshot, "interrupts", None) or ())]
            # the bridge may not see the __interrupt__ write yet (visible empty) ⇒ can't block, let it through
            if visible and self.interrupt_id not in visible:
                raise _error(
                    "This approval request has already been handled (interrupt_id is no longer valid)",
                    409,
                )
        return graph, config

    def _fb(self, trace_id: str | None) -> str | None:
        return feedback_token(self.principal.user_id, trace_id) if trace_id else None

    async def _result(self, root, messages: list, interrupts: Any) -> dict:
        """Result taken DIRECTLY from the graph output (no checkpoint re-read — see hitl.py)."""
        trace_id = tracing.current_trace_id()
        if interrupt := interrupt_payload(interrupts):
            if root is not None:
                root.update(output={"interrupt": interrupt})
            tracing.score_trace("hitl_requested", 1)
            return {
                "status": "interrupted",
                "interrupt": interrupt,
                "session_id": self.session_id,
                "trace_id": trace_id,
                "feedback_token": self._fb(trace_id),
            }
        answer, tools_used = _final_text(messages), _tools_used(messages)
        if root is not None:
            root.update(output={"response": answer, "tools_used": tools_used})
        return {
            "status": "success",
            "response": answer,
            "tools_used": tools_used,
            "session_id": self.session_id,
            "trace_id": trace_id,
            "feedback_token": self._fb(trace_id),
        }

    async def invoke(self) -> dict:
        async with _session_lock(self.principal.user_id, self.session_id):
            with self._trace([]) as root:
                try:
                    async with asyncio.timeout(self.settings.request_timeout_s):
                        graph, config = await self._prepare()
                        out = await graph.ainvoke(self.graph_input, config)
                except (
                    GreenNodeRequestError,
                    GraphRecursionError,
                    openai.APIError,
                    TimeoutError,
                ) as e:
                    mapped = _map_error(e)
                    if mapped is e:
                        raise
                    raise mapped from e
                return await self._result(root, out.get("messages", []), out.get("__interrupt__"))

    async def stream(self) -> AsyncIterator[dict]:
        lock = _session_lock(self.principal.user_id, self.session_id)
        await lock.acquire()
        it = None
        try:
            with (
                self._trace(["stream"]) as root,
                _deadline(self.settings.request_timeout_s) as left,
            ):
                async with asyncio.timeout(max(left(), 0.001)):
                    graph, config = await self._prepare()
                interrupts, final = None, None
                streaming: dict[
                    Any, str
                ] = {}  # langgraph_step → id of the agent message being streamed
                it = graph.astream(
                    self.graph_input, config, stream_mode=["messages", "updates", "values"]
                ).__aiter__()
                while True:
                    # Enforce the limit even when the graph is "silent" (LLM/Memory/MCP hung, no events emitted)
                    try:
                        mode, data = await asyncio.wait_for(it.__anext__(), timeout=max(left(), 0))
                    except StopAsyncIteration:
                        break
                    if mode == "values":
                        final = data
                        continue
                    if mode == "updates" and isinstance(data, dict) and data.get("__interrupt__"):
                        interrupts = data["__interrupt__"]
                    for event in _stream_events(mode, data, streaming):
                        yield event
                result = await self._result(root, (final or {}).get("messages", []), interrupts)
                if result["status"] == "interrupted":
                    yield {"event": "interrupt", **result}
                else:
                    yield {"event": "done", **result}
        except (GreenNodeRequestError, GraphRecursionError, openai.APIError, TimeoutError) as e:
            mapped = _map_error(e)
            yield {"event": "error", "message": mapped.message, "status": mapped.status_code}
        except Exception:  # noqa: BLE001 — stream already open, must return the error as an event
            log.exception("stream failed")
            yield {
                "event": "error",
                "message": "Agent failed to process the request.",
                "status": 500,
            }
        finally:
            if it is not None:
                with contextlib.suppress(Exception):
                    await it.aclose()
            lock.release()


@contextlib.contextmanager
def _deadline(seconds: float):
    """Countdown for the stream (an async generator can't be wrapped in asyncio.timeout)."""
    end = time.monotonic() + seconds
    yield lambda: end - time.monotonic()


def _stream_events(mode: str, data: Any, streaming: dict[Any, str] | None = None) -> list[dict]:
    """`streaming` (per request) remembers which message each agent step is streaming: a DIFFERENT message in the
    SAME step means the primary model failed mid-answer and a fallback model restarted it ⇒ `reset` first, so the
    client never shows "partial primary answer + fallback answer" glued together."""
    events: list[dict] = []
    if mode == "messages":
        chunk, meta = data
        if (
            isinstance(chunk, AIMessageChunk)
            and meta.get("langgraph_node") == "agent"
            and chunk.text
        ):
            if streaming is not None:
                step, previous = (
                    meta.get("langgraph_step"),
                    streaming.get(meta.get("langgraph_step")),
                )
                if previous is not None and chunk.id and chunk.id != previous:
                    events.append({"event": "reset", "reason": "llm_fallback"})
                if chunk.id:
                    streaming[step] = chunk.id
            events.append({"event": "token", "data": chunk.text})
        return events
    for node, update in (data or {}).items():
        if not isinstance(update, dict):
            continue
        if node == "reflect" and update.get("critique"):
            events.append({"event": "reset", "reason": "self_eval_retry"})
        for m in update.get("messages", []):
            if node == "agent" and isinstance(m, AIMessage):
                events.extend({"event": "tool_start", "name": tc["name"]} for tc in m.tool_calls)
            elif node == "tools" and isinstance(m, ToolMessage):
                events.append({"event": "tool_end", "name": m.name, "status": m.status})
    return events
