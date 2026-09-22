"""Per-user rate limiting when every request arrives through the frontend proxy.

Behind the Next.js proxy all traffic shares Vercel's egress addresses, so
keying on the peer alone would put every user in one bucket. The proxy
forwards the real client address in X-Client-IP; that header is honoured only
alongside a valid API key, because anyone can send it.
"""

from __future__ import annotations

import pytest

KEY = "proxy-secret"


@pytest.fixture
def limited(monkeypatch):
    """A minimal app carrying only the rate limiter, with an API key configured."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.rate_limit import RateLimitMiddleware
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "api_key", KEY)

    tiny = FastAPI()

    @tiny.get("/thing")
    def thing():
        return {"ok": True}

    tiny.add_middleware(RateLimitMiddleware, limit="3/minute", enabled=True)
    return TestClient(tiny)


def _statuses(client, n, headers_for):
    return [client.get("/thing", headers=headers_for(i)).status_code for i in range(n)]


def test_each_proxied_client_gets_its_own_budget(limited):
    a = {"X-API-Key": KEY, "X-Client-IP": "203.0.113.1"}
    b = {"X-API-Key": KEY, "X-Client-IP": "203.0.113.2"}
    assert _statuses(limited, 4, lambda i: a) == [200, 200, 200, 429]
    assert limited.get("/thing", headers=b).status_code == 200, "one user's traffic throttled another"


def test_client_ip_is_ignored_without_the_api_key(limited):
    statuses = _statuses(limited, 6, lambda i: {"X-Client-IP": f"203.0.113.{i}"})
    assert 429 in statuses, f"unauthenticated X-Client-IP reset the budget: {statuses}"


def test_client_ip_is_ignored_with_a_wrong_api_key(limited):
    statuses = _statuses(limited, 6, lambda i: {"X-API-Key": "guess", "X-Client-IP": f"203.0.113.{i}"})
    assert 429 in statuses, f"wrong key still let X-Client-IP reset the budget: {statuses}"


def test_client_ip_is_ignored_when_no_api_key_is_configured(limited, monkeypatch):
    """With no key there is no trusted proxy, so the header proves nothing."""
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "api_key", "")
    statuses = _statuses(limited, 6, lambda i: {"X-API-Key": "", "X-Client-IP": f"203.0.113.{i}"})
    assert 429 in statuses, f"X-Client-IP honoured with no key configured: {statuses}"


def test_malformed_client_ip_falls_back_to_the_peer_address(limited):
    statuses = _statuses(limited, 6, lambda i: {"X-API-Key": KEY, "X-Client-IP": f"not-an-ip-{i}"})
    assert 429 in statuses, f"arbitrary strings minted fresh buckets: {statuses}"
