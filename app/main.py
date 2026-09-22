"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.middleware import APIKeyMiddleware, RequestLoggingMiddleware, SecurityHeadersMiddleware, global_exception_handler
from app.api.rate_limit import RateLimitMiddleware
from app.api.routes import auth, chat, documents, health, feedback
from app.api.routes.admin import router as admin_router
from app.config import get_settings
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    settings = get_settings()
    setup_logging(settings.log_level)

    # SECRET_KEY is validated in app.config: it is a required field with no
    # in-source default, so a missing, placeholder, or too-short value fails
    # when Settings is constructed — before any request can be served.

    # APP_ENV drives the session cookie's Secure flag (see core/auth.py:
    # secure=not is_dev). A .env carried from a laptop to a server therefore
    # downgrades every session cookie to plaintext without changing any code
    # and without failing any test. Nothing here can detect "am I in
    # production", so this is a loud log line rather than a refusal to start.
    if settings.app_env.lower() in ("development", "dev", "local"):
        logger.warning(
            "insecure_session_cookies",
            app_env=settings.app_env,
            detail=(
                "Session cookies are being issued WITHOUT the Secure flag and "
                "will travel over plaintext HTTP. Correct for local work; set "
                "APP_ENV=production before exposing this service."
            ),
        )
    if not settings.api_key:
        logger.warning(
            "api_key_unset",
            detail=(
                "API_KEY is empty, so APIKeyMiddleware is a no-op and every "
                "route is reachable by anyone who can reach the port."
            ),
        )

    # Eagerly initialise the vector store so first request isn't slow
    from app.core.vector_store import get_vector_store

    get_vector_store()
    yield
    # Flush Langfuse traces on shutdown
    from app.core.observability import shutdown as langfuse_shutdown
    langfuse_shutdown()


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="RAG Chatbot API",
        description=(
            "Headless Retrieval-Augmented Generation API powered by Azure "
            "OpenAI GPT-5.2. JSON only — the UI is a separate client."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    # ── Rate limiter ──────────────────────────────────────────────────
    app.add_middleware(
        RateLimitMiddleware,
        limit=settings.rate_limit,
        enabled=settings.rate_limit_enabled,
    )

    # ── CORS ──────────────────────────────────────────────────────────
    origins = settings.cors_origin_list
    # Deny wildcard with credentials — that's a security misconfiguration
    if "*" in origins:
        origins = [o for o in origins if o != "*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "PATCH", "OPTIONS"],
        # X-User-Id deliberately absent: identity comes only from the signed
        # session cookie, never from a client-supplied header.
        allow_headers=["Content-Type", "Authorization", "Accept", "X-API-Key", "HX-Request", "HX-Target", "HX-Trigger"],
    )

    # ── Custom middleware ─────────────────────────────────────────────
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # ── Exception handlers ────────────────────────────────────────────
    app.add_exception_handler(Exception, global_exception_handler)

    # ── Routes ────────────────────────────────────────────────────────
    # Every route is JSON under /api/v1. There is no server-rendered page
    # router and no static mount: the client is a separate application that
    # talks to this API cross-origin, so CORS_ORIGINS is load-bearing rather
    # than incidental — an origin missing from it cannot sign in at all.
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(feedback.router, prefix="/api/v1")
    app.include_router(admin_router, prefix="/api/v1")

    return app


app = create_app()
