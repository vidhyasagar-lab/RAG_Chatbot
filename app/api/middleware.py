"""Custom exception handlers and middleware."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger

logger = get_logger(__name__)


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler so unhandled errors never leak stack traces."""
    logger.exception("unhandled_exception", path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject security headers into every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        # Note: 'unsafe-inline' is required by Tailwind CDN and HTMX runtime styles.
        # To remove it, switch to a self-hosted Tailwind build with nonce-based CSP.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://unpkg.com https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'"
        )
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request/response pair with timing."""

    async def dispatch(self, request: Request, call_next):
        import time

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed_ms=elapsed_ms,
        )
        return response


# ── Paths that bypass API-key authentication ─────────────────────────
# The browser-facing entry points only. /register sits alongside /login —
# omitting it meant enabling API_KEY silently broke signup but not sign-in.
# API docs are deliberately NOT public: when API_KEY is set, publishing the
# full request surface unauthenticated defeats the point of having a key.
_PUBLIC_PATHS = frozenset({
    "/",
    "/login",
    "/register",
    "/logout",
    "/api/v1/health",
})
_PUBLIC_PREFIXES = ("/static/", "/partials/")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require a valid X-API-Key header when ``API_KEY`` is configured.

    If ``settings.api_key`` is empty the middleware is a no-op, keeping
    local development friction-free.
    """

    async def dispatch(self, request: Request, call_next):
        from app.config import get_settings

        settings = get_settings()

        # Skip auth if no key configured or path is public
        if (
            not settings.api_key
            or request.url.path in _PUBLIC_PATHS
            or any(request.url.path.startswith(p) for p in _PUBLIC_PREFIXES)
        ):
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        import hmac
        if not hmac.compare_digest(provided, settings.api_key):
            logger.warning("auth_failed", path=request.url.path)
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )
        return await call_next(request)
