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


# ── X-Forwarded-For, as the real stack presents it ───────────────────
#
# The app runs behind Caddy with `--proxy-headers --forwarded-allow-ips "*"`.
# That makes uvicorn rewrite scope["client"] from X-Forwarded-For, and with
# every host trusted it takes the **leftmost** entry:
#
#     if self.always_trust:
#         return _parse_host_port(x_forwarded_for_hosts[0])
#
# Caddy appends the true client address, so a caller who sends their own
# X-Forwarded-For puts a value of their choosing to the left of it - and
# request.client.host, which the limiter keyed on, became theirs to pick.
#
# The published port is bound to 127.0.0.1, so the only hop that can reach
# this app is Caddy on the same host. The entry Caddy appends is therefore
# the rightmost one, and the one to trust.


def _proxied(monkeypatch, trusted_hosts):
    """The app wrapped the way uvicorn wraps it, with a given trust list."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    from app.api.rate_limit import RateLimitMiddleware
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "api_key", KEY)

    tiny = FastAPI()

    @tiny.get("/thing")
    def thing():
        return {"ok": True}

    tiny.add_middleware(RateLimitMiddleware, limit="3/minute", enabled=True)
    return TestClient(ProxyHeadersMiddleware(tiny, trusted_hosts=trusted_hosts))


@pytest.fixture
def behind_proxy(monkeypatch):
    """Wrapped as production wraps it: the real trust list.

    TestClient's peer is the literal "testclient", which is not in the list,
    so uvicorn treats this app as directly exposed and leaves the client
    alone - exactly how it behaves for a caller who reaches the port without
    going through Caddy.
    """
    from app.api.rate_limit import TRUSTED_PROXY_IPS

    return _proxied(monkeypatch, TRUSTED_PROXY_IPS)


@pytest.fixture
def behind_trusting_proxy(monkeypatch):
    """Wrapped the way it *was*: every host trusted. Kept to prove the
    bypass is a property of that setting and not of the limiter."""
    return _proxied(monkeypatch, "*")


def test_trusting_every_host_is_what_opened_the_bypass(behind_trusting_proxy):
    """The original defect, pinned so the reason is not lost. With "*"
    uvicorn takes the leftmost forwarded entry, which the caller controls."""
    statuses = [
        behind_trusting_proxy.get(
            "/thing", headers={"X-Forwarded-For": f"10.0.0.{i}, 198.51.100.7"}
        ).status_code
        for i in range(6)
    ]

    assert statuses.count(200) == 6, (
        "this test documents the old behaviour; if it now throttles, the "
        f"explanation above is stale: {statuses}"
    )


def test_the_deployment_does_not_trust_every_host():
    """The limiter's correctness rests on this flag, and nothing in the
    application would fail if someone set it back to "*"."""
    from pathlib import Path

    from app.api.rate_limit import TRUSTED_PROXY_IPS

    dockerfile = Path(__file__).resolve().parents[1] / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")

    assert '--forwarded-allow-ips", "*"' not in text, (
        "trusting every host makes uvicorn read the leftmost X-Forwarded-For "
        "entry, which the caller chooses"
    )
    assert TRUSTED_PROXY_IPS in text, (
        f"Dockerfile must pass --forwarded-allow-ips {TRUSTED_PROXY_IPS}"
    )


@pytest.mark.parametrize("forwarded", [
    "10.0.0.{i}, 198.51.100.7",   # a spoof to the left of the appended address
    "198.51.100.{i}",             # a bare invented address
    "garbage-{i}",                # not an address at all
    "10.0.0.{i}, 10.0.0.{i}",     # every hop claimed to be private
])
def test_no_forwarded_header_can_mint_fresh_buckets(behind_proxy, forwarded):
    """Whatever the caller puts in X-Forwarded-For, the budget is spent.

    With the real trust list this app's peer is untrusted, so uvicorn leaves
    the client address alone and the header is ignored entirely.
    """
    statuses = [
        behind_proxy.get("/thing", headers={"X-Forwarded-For": forwarded.format(i=i)}).status_code
        for i in range(6)
    ]

    assert 429 in statuses, f"{forwarded!r} reset the budget each request: {statuses}"


def test_a_spoofed_entry_does_not_drain_another_clients_budget(behind_proxy):
    """The mirror image: naming someone else must not spend their budget."""
    for _ in range(4):
        behind_proxy.get("/thing", headers={"X-Forwarded-For": "198.51.100.9"})

    victim = behind_proxy.get("/thing", headers={"X-API-Key": KEY, "X-Client-IP": "198.51.100.9"})

    assert victim.status_code == 200, "a spoofed entry drained the named client's budget"


def test_a_real_proxy_hop_resolves_to_the_appended_client():
    """The other half: when the peer *is* a trusted proxy, uvicorn must
    resolve the client to the address that proxy appended - not the one the
    caller prepended, and not the proxy itself.

    Asserted against uvicorn's own resolver, because TestClient cannot
    present a private peer address.
    """
    from uvicorn.middleware.proxy_headers import _TrustedHosts

    from app.api.rate_limit import TRUSTED_PROXY_IPS

    hosts = _TrustedHosts(TRUSTED_PROXY_IPS)

    # Caller prepended a lie; Caddy appended the truth.
    assert hosts.get_trusted_client_address("198.51.100.99, 203.0.113.5")[0] == "203.0.113.5"
    # Caddy overwrote instead of appending.
    assert hosts.get_trusted_client_address("203.0.113.5")[0] == "203.0.113.5"
    # A private hop in the chain is skipped, not reported as the client.
    assert hosts.get_trusted_client_address("203.0.113.5, 172.18.0.1")[0] == "203.0.113.5"


def test_the_trusted_proxy_header_still_wins(behind_proxy):
    """Vercel's proxy holds the API key and names the user in X-Client-IP.
    That path must keep working, since behind it every user shares one
    egress address."""
    base = {"X-API-Key": KEY, "X-Forwarded-For": "198.51.100.7"}
    a = {**base, "X-Client-IP": "203.0.113.1"}
    b = {**base, "X-Client-IP": "203.0.113.2"}

    assert [behind_proxy.get("/thing", headers=a).status_code for _ in range(4)] == [200, 200, 200, 429]
    assert behind_proxy.get("/thing", headers=b).status_code == 200
