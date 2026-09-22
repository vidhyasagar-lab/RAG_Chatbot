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
        # Explicitly disabled rather than "1; mode=block". The XSS Auditor is
        # gone from every current browser, and where it survives its filtering
        # has itself been used to leak cross-origin data. "0" is the value
        # current guidance recommends; CSP below is the real control.
        response.headers["X-XSS-Protection"] = "0"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

        # This service returns JSON and nothing else, so the policy denies
        # everything by default. The previous policy allowed 'unsafe-inline'
        # scripts plus cdn.tailwindcss.com, unpkg.com and cdn.jsdelivr.net,
        # which the Jinja/HTMX UI needed; that UI is gone, and leaving its
        # allowances in place would silently permit script execution on any
        # HTML this service ever returned by accident.
        #
        # The exception is FastAPI's own docs pages, which are real HTML and
        # load Swagger/ReDoc bundles from jsdelivr. They are scoped to exactly
        # those paths rather than relaxing the policy everywhere.
        if request.url.path in ("/docs", "/redoc", "/docs/oauth2-redirect"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'none'; "
                "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
                "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
                "img-src 'self' https://fastapi.tiangolo.com data:; "
                "connect-src 'self'; "
                "font-src 'self' https://cdn.jsdelivr.net; "
                "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
            )
        else:
            response.headers["Content-Security-Policy"] = (
                "default-src 'none'; "
                "base-uri 'none'; "
                "form-action 'none'; "
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
# The unauthenticated entry points only. /auth/register sits alongside
# /auth/login — omitting it meant enabling API_KEY silently broke signup but
# not sign-in. /auth/me is deliberately absent: it reports who you are, which
# is not something an unauthenticated caller needs.
# API docs are deliberately NOT public: when API_KEY is set, publishing the
# full request surface unauthenticated defeats the point of having a key.
_PUBLIC_PATHS = frozenset({
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/api/v1/auth/logout",
    "/api/v1/health",
})
# No public prefixes: /static/ and /partials/ served the server-rendered UI,
# which no longer exists. An empty tuple keeps the any() check below honest
# rather than leaving dead prefixes that would silently exempt future routes
# happening to live under those paths.
_PUBLIC_PREFIXES: tuple[str, ...] = ()


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
