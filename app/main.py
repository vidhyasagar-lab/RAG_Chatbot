"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.middleware import APIKeyMiddleware, RequestLoggingMiddleware, SecurityHeadersMiddleware, global_exception_handler
from app.api.rate_limit import RateLimitMiddleware
from app.api.routes import chat, documents, health, feedback
from app.api.routes.admin import router as admin_router
from app.api.routes.pages import router as pages_router
from app.config import DEFAULT_SECRET_KEY as _DEFAULT_SECRET_KEY, get_settings
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    settings = get_settings()
    setup_logging(settings.log_level)

    # Refuse to serve production traffic with the shipped placeholder secret —
    # session cookies signed with a known key are trivially forgeable.
    if settings.secret_key == _DEFAULT_SECRET_KEY:
        if settings.app_env.lower() in ("development", "dev", "local"):
            logger.warning("secret_key_is_default_dev_only")
        else:
            raise RuntimeError(
                "SECRET_KEY is still the shipped default. Set SECRET_KEY in .env to a "
                "random value (e.g. `python -c \"import secrets;print(secrets.token_urlsafe(32))\"`)."
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
        description="Production-grade Retrieval-Augmented Generation chatbot powered by Azure OpenAI GPT-5.2",
        version="1.0.0",
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
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(feedback.router, prefix="/api/v1")
    app.include_router(admin_router, prefix="/api/v1")

    # ── Server-rendered pages (Jinja2 + HTMX) ────────────────────────
    app.include_router(pages_router)

    # ── Static files ──────────────────────────────────────────────────
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


app = create_app()
