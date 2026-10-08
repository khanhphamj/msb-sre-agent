"""A2A (Agent2Agent) server — lets OTHER agents call this agent via the A2A standard (a2a-sdk 1.x, JSON-RPC).

Routes (same container :8080, next to /invocations):
  GET  /.well-known/agent-card.json   Agent Card (public — describes skills + how to authenticate)
  POST /a2a                           JSON-RPC: SendMessage, GetTask, ListTasks, CancelTask...

Authentication & per-user isolation (same as /invocations):
  - Every /a2a request goes through `authenticate()` (AUTH_MODE jwt | api_key) ⇒ 401 if invalid.
  - The authenticated user is attached to `request.user` ⇒ the a2a-sdk task store keys tasks by owner =
    user_id ⇒ user B cannot GetTask/ListTasks user A's tasks.
  - A2A `contextId` ⇒ session_id `a2a-<contextId>`; memory is keyed by (session, user) like every other flow.
  - HITL: graph pauses ⇒ task `INPUT_REQUIRED` + description of the tool awaiting approval; the caller sends
    the next message on the same contextId with "approve" or "reject: <reason>" to continue.
"""

from __future__ import annotations

import logging
import uuid

from a2a.helpers.proto_helpers import new_task, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue_v2 import EventQueue
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import a2a_pb2 as pb
from greennode_agentbase import RequestContext as AgentBaseRequestContext
from greennode_agentbase.exceptions import GreenNodeRequestError
from starlette.applications import Starlette
from starlette.authentication import (
    AuthCredentials,
    AuthenticationBackend,
    AuthenticationError,
    SimpleUser,
)
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from starlette.routing import BaseRoute, Mount

from app.a2a.decisions import parse_decision
from app.auth.inbound import authenticate_async, validate_session_id
from app.config import Settings, get_settings

log = logging.getLogger(__name__)

RPC_PATH = "/a2a"


# --------------------------------------------------------------------------- auth
class AgentBaseAuthBackend(AuthenticationBackend):
    """Reuses the agent's inbound auth for A2A (same headers as /invocations)."""

    async def authenticate(self, conn: HTTPConnection):
        headers = dict(conn.headers)
        ctx = AgentBaseRequestContext(
            session_id=None,
            user_id=conn.headers.get("x-greennode-agentbase-user-id"),
            request_headers=headers,
        )
        try:
            principal = await authenticate_async(ctx, get_settings())
        except GreenNodeRequestError as e:
            raise AuthenticationError(e.message) from e
        return AuthCredentials(["authenticated"]), SimpleUser(principal.user_id)


def _auth_error(_conn: HTTPConnection, exc: Exception) -> JSONResponse:
    return JSONResponse({"error": str(exc)}, status_code=401)


# --------------------------------------------------------------------------- executor
def _text_of(message: pb.Message | None) -> str:
    return "\n".join(p.text for p in (message.parts if message else []) if p.text).strip()


class AgentBaseExecutor(AgentExecutor):
    """Wraps the standard graph (service.run_chat / run_resume) as an A2A task."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        from app import service  # avoid a circular import at app init

        user_id = context.call_context.user.user_name
        session_id = f"a2a-{context.context_id}"
        if (
            context.current_task is None
        ):  # A2A v1: the Task must be enqueued before any status update
            await event_queue.enqueue_event(
                new_task(
                    context.task_id,
                    context.context_id,
                    pb.TASK_STATE_SUBMITTED,
                    history=[context.message] if context.message else None,
                )
            )
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        text = _text_of(context.message)
        await updater.start_work()
        try:
            validate_session_id(
                session_id
            )  # contextId ends up in the Memory path (prevent traversal)
            decision = parse_decision(text)
            pending = (
                await service.pending_tool_calls(user_id=user_id, session_id=session_id)
                if decision
                else []
            )
            if pending:
                approve, reason = decision
                decisions = [
                    {
                        "tool_call_id": tc["id"],
                        "action": "approve" if approve else "reject",
                        "reason": None if approve else (reason or "Rejected via A2A"),
                    }
                    for tc in pending
                ]
                result = await service.run_resume(decisions, user_id=user_id, session_id=session_id)
            else:
                result = await service.run_chat(text, user_id=user_id, session_id=session_id)
        except GreenNodeRequestError as e:
            if e.status_code == 409:  # approval pending but the caller sent something else
                await updater.requires_input(
                    updater.new_agent_message(
                        [
                            pb.Part(
                                text=(
                                    "An action is awaiting confirmation — reply 'approve' or 'reject: <reason>'."
                                )
                            )
                        ]
                    )
                )
                return
            await updater.failed(updater.new_agent_message([pb.Part(text=e.message)]))
            return

        if result["status"] == "interrupted":
            calls = ", ".join(
                f"{t['name']}({t['args']})" for t in result["interrupt"]["tool_calls"]
            )
            await updater.requires_input(
                updater.new_agent_message(
                    [
                        pb.Part(
                            text=f"Confirmation required before running: {calls}. "
                            "Reply 'approve' or 'reject: <reason>'."
                        )
                    ]
                )
            )
            return
        await updater.add_artifact([pb.Part(text=result["response"])], name="response")
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()


# --------------------------------------------------------------------------- card
def build_agent_card(settings: Settings) -> pb.AgentCard:
    base = settings.a2a_public_url.rstrip("/")
    if settings.auth_mode == "api_key":
        schemes = {
            "apiKey": pb.SecurityScheme(
                api_key_security_scheme=pb.APIKeySecurityScheme(
                    location="header",
                    name=settings.auth_api_key_header,
                    description="API key; send header X-GreenNode-AgentBase-User-Id for per-user memory",
                )
            )
        }
    else:
        schemes = {
            "bearer": pb.SecurityScheme(
                http_auth_security_scheme=pb.HTTPAuthSecurityScheme(
                    scheme="bearer",
                    bearer_format="JWT",
                    description=f"JWT issued by {settings.auth_issuer}",
                )
            )
        }
    return pb.AgentCard(
        name=settings.agent_name,
        description=settings.a2a_description,
        version=settings.agent_version,
        supported_interfaces=[
            pb.AgentInterface(url=f"{base}{RPC_PATH}", protocol_binding="JSONRPC")
        ],
        capabilities=pb.AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        security_schemes=schemes,
        skills=[
            pb.AgentSkill(
                id=s["id"],
                name=s["name"],
                description=s["description"],
                tags=s.get("tags", []),
                examples=s.get("examples", []),
            )
            for s in settings.a2a_skills
        ],
    )


def a2a_routes(settings: Settings) -> list[BaseRoute]:
    card = build_agent_card(settings)
    handler = DefaultRequestHandlerV2(
        agent_executor=AgentBaseExecutor(), task_store=InMemoryTaskStore(), agent_card=card
    )
    # Sub-app holds only the JSON-RPC route at /a2a, wrapped in AuthenticationMiddleware. Mount("") is
    # placed AFTER /invocations, /health, agent card ⇒ those routes are unaffected.
    rpc_app = Starlette(
        routes=create_jsonrpc_routes(handler, rpc_url=RPC_PATH),
        middleware=[
            Middleware(
                AuthenticationMiddleware, backend=AgentBaseAuthBackend(), on_error=_auth_error
            )
        ],
    )
    return [*create_agent_card_routes(card), Mount("", app=rpc_app)]


def new_request_id() -> str:
    return uuid.uuid4().hex


__all__ = ["a2a_routes", "build_agent_card", "new_text_message"]
