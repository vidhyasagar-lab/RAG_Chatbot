"""Regression tests for the security findings fixed in the audit.

Each test names the finding it guards so a future change that reopens one
fails loudly rather than silently.
"""

from __future__ import annotations

import pytest


# ── SEC-2: the passwordless login endpoint is gone ───────────────────────

def test_passwordless_login_endpoint_removed(client):
    """It returned any existing user's id and role for a bare username."""
    resp = client.post("/api/v1/users/login", json={"username": "anyone"})
    assert resp.status_code == 404, (
        "POST /api/v1/users/login is reachable again - it authenticated "
        "with no password and leaked user ids"
    )


def test_users_router_not_registered(client):
    schema = client.get("/openapi.json").json()
    assert not [p for p in schema["paths"] if p.startswith("/api/v1/users")]


# ── SEC-4: endpoints that were unauthenticated ───────────────────────────

@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/v1/documents/stats"),
        ("get", "/api/v1/chat/scores/some-trace-id"),
        ("post", "/api/v1/feedback/"),
    ],
)
def test_endpoints_require_authentication(client, method, path):
    kwargs = {"json": {"trace_id": "t", "score": 1}} if method == "post" else {}
    resp = getattr(client, method)(path, **kwargs)
    assert resp.status_code == 401, f"{path} answered {resp.status_code} unauthenticated"


# ── SEC-1: rate limiting is actually enforced ────────────────────────────

def test_rate_limit_middleware_is_registered(client):
    from app.api.rate_limit import RateLimitMiddleware
    from app.main import app

    registered = [m.cls for m in app.user_middleware]
    assert RateLimitMiddleware in registered, "no rate limiting in the stack"


def _limited_client(limit="5/minute"):
    """A minimal app carrying only the rate limiter, for isolated testing."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.rate_limit import RateLimitMiddleware

    tiny = FastAPI()

    @tiny.get("/thing")
    def thing():
        return {"ok": True}

    @tiny.get("/api/v1/health")
    def health():
        return {"status": "healthy"}

    tiny.add_middleware(RateLimitMiddleware, limit=limit, enabled=True)
    return TestClient(tiny)


def test_rate_limit_returns_429_when_exceeded():
    """Requests past the budget must be rejected.

    Guards the original bug *and* its subtler second half: slowapi exempts any
    request whose route handler it cannot resolve, and current FastAPI hides
    included routes behind ``_IncludedRouter`` — so simply registering
    SlowAPIMiddleware limited nothing at all. This asserts observed behaviour,
    never mere registration.
    """
    c = _limited_client("5/minute")
    statuses = [c.get("/thing").status_code for _ in range(8)]
    assert statuses[:5] == [200] * 5, f"limit kicked in too early: {statuses}"
    assert 429 in statuses[5:], f"no 429 after exceeding 5/minute: {statuses}"


def test_rate_limited_response_carries_retry_after():
    c = _limited_client("2/minute")
    for _ in range(3):
        resp = c.get("/thing")
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


def test_health_endpoint_is_exempt_from_rate_limiting():
    """Liveness probes must not be throttled into looking unhealthy."""
    c = _limited_client("3/minute")
    statuses = {c.get("/api/v1/health").status_code for _ in range(10)}
    assert statuses == {200}, f"health check got throttled: {statuses}"


def test_rate_limit_can_be_disabled():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.rate_limit import RateLimitMiddleware

    tiny = FastAPI()

    @tiny.get("/thing")
    def thing():
        return {"ok": True}

    tiny.add_middleware(RateLimitMiddleware, limit="2/minute", enabled=False)
    c = TestClient(tiny)
    assert {c.get("/thing").status_code for _ in range(6)} == {200}


def test_forwarded_header_cannot_reset_the_budget():
    """X-Forwarded-For is client-controlled; honouring it would defeat the limit."""
    c = _limited_client("3/minute")
    statuses = []
    for i in range(8):
        statuses.append(
            c.get("/thing", headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code
        )
    assert 429 in statuses, f"spoofed forwarding header reset the budget: {statuses}"


# ── HYG-2 / HYG-3: API-key public path list ──────────────────────────────

def test_register_is_public_alongside_login():
    from app.api.middleware import _PUBLIC_PATHS

    assert "/api/v1/auth/register" in _PUBLIC_PATHS, (
        "enabling API_KEY would break signup while login kept working"
    )


def test_auth_me_is_not_public():
    """/auth/me reports identity, so it must not bypass the API key."""
    from app.api.middleware import _PUBLIC_PATHS

    assert "/api/v1/auth/me" not in _PUBLIC_PATHS


def test_api_docs_are_not_public():
    from app.api.middleware import _PUBLIC_PATHS

    for path in ("/docs", "/redoc", "/openapi.json"):
        assert path not in _PUBLIC_PATHS, (
            f"{path} bypasses the API key, publishing the full API surface"
        )


# ── HYG-5: password comparison is constant time ──────────────────────────

def test_password_verification_is_constant_time():
    import inspect

    from app.core import user_store

    src = inspect.getsource(user_store._verify_password)
    assert "compare_digest" in src, "digest compared with == leaks timing"


def test_password_roundtrip_and_rejection():
    from app.core.user_store import _hash_password, _verify_password

    stored = _hash_password("correct-horse-battery")
    assert _verify_password("correct-horse-battery", stored) is True
    assert _verify_password("wrong-password", stored) is False
    assert _verify_password("anything", "") is False
    assert _verify_password("anything", "not:hex") is False


# ── HYG-6: secrets come from .env only, and weak ones are rejected ───────

def _settings_with(secret: str):
    """Build Settings with an explicit secret_key, ignoring the ambient .env."""
    from app.config import Settings

    return Settings(
        _env_file=None,
        azure_openai_api_key="k",
        azure_openai_endpoint="https://example.openai.azure.com",
        secret_key=secret,
    )


def test_config_has_no_hardcoded_secret_key_default(monkeypatch):
    """secret_key must be required — an in-source default is a known signing key."""
    import pydantic

    from app.config import Settings

    field = Settings.model_fields["secret_key"]
    assert field.is_required(), "secret_key must not have a default in config.py"

    # conftest exports SECRET_KEY for the rest of the suite; drop it here so the
    # absence of an in-source default is what the assertion actually observes.
    monkeypatch.delenv("SECRET_KEY", raising=False)
    with pytest.raises(pydantic.ValidationError):
        Settings(
            _env_file=None,
            azure_openai_api_key="k",
            azure_openai_endpoint="https://example.openai.azure.com",
        )


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "change-me-to-a-random-secret", "CHANGE-ME-TO-A-RANDOM-SECRET", "secret", "short"],
)
def test_weak_secret_keys_are_rejected(bad):
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        _settings_with(bad)


def test_strong_secret_key_is_accepted():
    import secrets

    good = secrets.token_urlsafe(32)
    assert _settings_with(good).secret_key == good


def test_no_real_secret_values_live_outside_env():
    """Secret-shaped settings must default to empty, never to a usable value."""
    from app.config import Settings

    for name in (
        "api_key",
        "langfuse_public_key",
        "langfuse_secret_key",
        "azure_openai_embedding_api_key",
    ):
        field = Settings.model_fields[name]
        assert field.default == "", f"{name} must default to empty, not a real value"

    for name in ("azure_openai_api_key", "azure_openai_endpoint", "secret_key"):
        assert Settings.model_fields[name].is_required(), f"{name} must come from .env"


# ── DEAD-1: the legacy standalone UI is gone ─────────────────────────────

def test_legacy_static_ui_removed():
    from pathlib import Path

    import app.main as main_mod

    root = Path(main_mod.__file__).resolve().parent.parent
    assert not (root / "static" / "index.html").exists(), (
        "static/index.html is back: it bypassed the API-key gate and "
        "authenticated through the removed passwordless endpoint"
    )


def test_no_server_rendered_ui_is_mounted():
    """The API serves no HTML of its own.

    Guards the headless boundary from both directions: no templates on disk
    and no StaticFiles mount in the app. Either one returning would mean the
    server is rendering markup again, which is what the XSS controls below
    used to exist for.
    """
    from pathlib import Path

    from starlette.staticfiles import StaticFiles

    import app.main as main_mod
    from app.main import app

    root = Path(main_mod.__file__).resolve().parent.parent
    assert not (root / "templates").exists(), "templates/ is back"

    mounted = [r for r in app.routes if isinstance(getattr(r, "app", None), StaticFiles)]
    assert not mounted, f"a StaticFiles mount is registered: {mounted}"


# ── SEC-3: model output sanitisation now belongs to the client ───────────
#
# Two tests were deleted here, and the guarantee they enforced did not move
# somewhere else in this repo - it left it.
#
#   test_markdown_output_is_sanitised  asserted that app.html ran model
#       output through DOMPurify.sanitize before assigning innerHTML.
#   test_cdn_scripts_are_version_pinned  asserted base.html pinned its
#       jsdelivr/unpkg script tags to a version.
#
# Both read templates/ that no longer exists. The RAG engine still returns
# model-authored markdown, and rendering it with innerHTML is still an XSS
# sink - but the renderer is now a separate client application, which this
# suite cannot see. Whatever consumes /api/v1/chat MUST sanitise before
# rendering; nothing on the server side will catch it if it does not.
