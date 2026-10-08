"""AgentBase Runtime entrypoint. KEEP THIS FILE THIN — logic lives in app/.

Platform contract: listen on 0.0.0.0:8080, GET /health returns 200.
The SDK provides POST /invocations; the handler is async and returns a dict (JSON) or an async generator (SSE).
"""

from __future__ import annotations

import contextlib
import logging

from dotenv import load_dotenv

load_dotenv()

from greennode_agentbase import GreenNodeAgentBaseApp, PingStatus, RequestContext  # noqa: E402
from greennode_agentbase.runtime.app import XAccelBufferingMiddleware  # noqa: E402
from starlette.middleware import Middleware  # noqa: E402
from starlette.middleware.cors import CORSMiddleware  # noqa: E402

from app import service  # noqa: E402
from app.auth.inbound import prefetch_jwks  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.observability import tracing  # noqa: E402

settings = get_settings()
logging.basicConfig(level=settings.log_level)


@contextlib.asynccontextmanager
async def lifespan(_app):
    tracing.init_tracing(settings)
    await prefetch_jwks(settings)  # the first request doesn't wait for the JWKS download
    try:
        yield
    finally:
        if settings.zalo_enabled:
            from app.channels.zalo import shutdown_zalo

            await shutdown_zalo()
        tracing.shutdown_tracing()


# Passing middleware => the SDK drops its default middleware, so XAccelBufferingMiddleware (SSE) must be re-added.
middleware = [Middleware(XAccelBufferingMiddleware)]
if settings.cors_allow_origins:
    middleware.append(
        Middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_methods=["POST", "GET", "OPTIONS"],
            allow_headers=["*"],
        )
    )

app = GreenNodeAgentBaseApp(lifespan=lifespan, middleware=middleware)

if (
    settings.a2a_enabled
):  # A2A server: /.well-known/agent-card.json + POST /a2a (xem app/a2a/server.py)
    from app.a2a.server import a2a_routes

    app.router.routes.extend(a2a_routes(settings))

if settings.zalo_enabled:  # Zalo Bot webhook: POST /webhook/zalo (see app/channels/zalo.py)
    from app.channels.zalo import zalo_routes

    app.router.routes.extend(zalo_routes(settings))


@app.entrypoint
async def handler(payload: dict, context: RequestContext):
    return await service.handle(payload, context)


@app.ping
def health_check() -> PingStatus:
    return PingStatus.HEALTHY


if __name__ == "__main__":
    app.run(port=8080, host="0.0.0.0")
