"""Per-IP request rate limiting.

Why this is hand-rolled rather than ``slowapi``'s middleware: slowapi resolves
the matched route handler with ``_find_route_handler(app.routes, scope)`` and
**exempts the request when it cannot find one**. Current FastAPI wraps included
routers in an internal ``_IncludedRouter`` object that exposes no ``.endpoint``,
so every route registered via ``include_router`` — which is all of them —
resolved to ``None`` and was silently skipped. Registering
``SlowAPIMiddleware`` therefore had no effect at all.

This middleware keys off the client address only, so it never depends on route
introspection and cannot be quietly disabled by a framework internal changing
shape.
"""

from __future__ import annotations

from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)

# Paths that must stay reachable regardless of budget: liveness probes should
# report health rather than being throttled into looking unhealthy.
_EXEMPT_PATHS = frozenset({"/api/v1/health"})
_EXEMPT_PREFIXES = ("/static/",)


def _client_key(request) -> str:
    """Bucket key for a request.

    Deliberately the *direct* peer address. ``X-Forwarded-For`` is client
    controlled and trivially spoofed, so honouring it would let anyone reset
    their own budget. Behind a trusted proxy, configure the proxy to set the
    peer address instead (uvicorn ``--proxy-headers``).
    """
    client = request.client
    return client.host if client and client.host else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests from an address that exceeds the configured budget."""

    def __init__(self, app, limit: str = "60/minute", enabled: bool = True) -> None:
        super().__init__(app)
        self.enabled = enabled
        self.storage = MemoryStorage()
        self.strategy = MovingWindowRateLimiter(self.storage)
        self.limit = parse(limit)
        self.limit_text = limit

    async def dispatch(self, request, call_next):
        if not self.enabled:
            return await call_next(request)

        path = request.url.path
        if path in _EXEMPT_PATHS or path.startswith(_EXEMPT_PREFIXES):
            return await call_next(request)

        key = _client_key(request)
        if not self.strategy.hit(self.limit, key):
            reset_at, _remaining = self.strategy.get_window_stats(self.limit, key)
            logger.warning("rate_limit_exceeded", client=key, path=path, limit=self.limit_text)
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded ({self.limit_text})"},
                headers={"Retry-After": str(max(1, int(reset_at)))},
            )

        response = await call_next(request)
        _reset, remaining = self.strategy.get_window_stats(self.limit, key)
        response.headers["X-RateLimit-Limit"] = str(self.limit.amount)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        return response
