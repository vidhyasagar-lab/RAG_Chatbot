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

import hmac
import ipaddress

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


def _from_trusted_proxy(request) -> bool:
    """True when the request carries the configured API key.

    Only the frontend's server-side proxy holds the key, so its headers can be
    believed. With no key configured nothing is trusted.
    """
    from app.config import get_settings

    key = get_settings().api_key
    if not key:
        return False
    provided = request.headers.get("X-API-Key", "")
    return hmac.compare_digest(provided.encode(), key.encode())


#: What ``--forwarded-allow-ips`` must be set to, and why the peer address
#: below can be believed at all.
#:
#: uvicorn runs with ``--proxy-headers``, so it rewrites ``scope["client"]``
#: from ``X-Forwarded-For`` - and *which* entry it takes depends entirely on
#: this list. Trusting every host (``"*"``) made it take the **leftmost**::
#:
#:     if self.always_trust:
#:         return _parse_host_port(x_forwarded_for_hosts[0])
#:
#: Caddy appends the true client address, so a caller who sent their own
#: ``X-Forwarded-For`` placed a value of their choosing to the left of it and
#: ``request.client.host`` became theirs to pick - a fresh rate-limit bucket
#: on every request. With a real list uvicorn instead walks the header from
#: the right and returns the first *untrusted* hop, which is the address
#: Caddy appended.
#:
#: The whole private range rather than one gateway address: the app reaches
#: this host through Docker's bridge, and Compose picks that subnet itself
#: (172.17.0.1, 172.18.0.1, ...). Naming one would break silently - uvicorn
#: would stop rewriting, every user would land in the gateway's bucket, and
#: one person's traffic would throttle everyone. A public client address is
#: never private, so nothing real is skipped.
#:
#: ``tests/test_rate_limit_proxy.py`` pins the Dockerfile to this value.
TRUSTED_PROXY_IPS = "127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"


def _client_key(request) -> str:
    """Bucket key for a request.

    The *direct* peer address, which uvicorn has resolved to the real client
    using ``TRUSTED_PROXY_IPS``. ``X-Forwarded-For`` is deliberately not read
    here: the app cannot tell whether it arrived through the reverse proxy or
    straight from a caller who made it up, and trusting it unconditionally
    hands every caller a fresh budget whenever the port is reachable.

    The one exception is ``X-Client-IP`` on a request that also carries a
    valid API key. Only the frontend's server-side proxy holds the key, and
    behind it every user shares one egress address, so without this they
    would all share one bucket.
    """
    client = request.client
    peer = client.host if client and client.host else "unknown"
    named = request.headers.get("X-Client-IP")
    if named and _from_trusted_proxy(request):
        try:
            return str(ipaddress.ip_address(named.strip()))
        except ValueError:
            logger.warning("rate_limit_bad_client_ip", peer=peer)
    return peer


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
