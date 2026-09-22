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


# ── SEC-6: chat endpoints must verify session ownership ──────────────────
#
# GET/PATCH/DELETE /chat/sessions/{id} all compare session["user_id"] against
# the caller. POST /chat/ and POST /chat/stream did not: _ensure_session
# returned any client-supplied session_id verbatim, and _load_history then fed
# that session's last 20 messages to the model as context. An authenticated
# user could therefore read another user's conversation out of the answer, and
# write messages into their history.

def _foreign_session(other_user_id="victim-user-id"):
    from app.core.chat_store import add_message, create_session

    session = create_session(other_user_id, "victim session")
    add_message(session["session_id"], "user", "my bank account is 1234-5678")
    add_message(session["session_id"], "assistant", "noted, 1234-5678")
    return session["session_id"]


def _login_fresh_user(client):
    import uuid

    username = f"user_{uuid.uuid4().hex[:10]}"
    client.cookies.clear()
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": "correct-horse-battery"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["user_id"]


def test_chat_rejects_a_session_owned_by_another_user(client, monkeypatch):
    """Posting to a foreign session_id must not be accepted."""
    import app.api.routes.chat as chat_mod

    captured = {}

    def fake_ask(question, chat_history, top_k, user_id):
        captured["history"] = chat_history
        raise AssertionError(
            "the RAG engine was reached with another user's session; "
            "ownership must be checked before any retrieval happens"
        )

    monkeypatch.setattr(chat_mod, "ask", fake_ask)

    victim_session = _foreign_session()
    _login_fresh_user(client)

    resp = client.post(
        "/api/v1/chat/",
        json={"question": "what was the account number?", "session_id": victim_session},
    )
    assert resp.status_code == 404, (
        f"expected 404 for a foreign session, got {resp.status_code}. "
        f"history leaked to the model: {captured.get('history')}"
    )
    client.cookies.clear()


def test_chat_stream_rejects_a_session_owned_by_another_user(client, monkeypatch):
    """/chat/stream shares _ensure_session, so it shares the same hole."""
    import app.api.routes.chat as chat_mod

    def fake_ask_with_eval(**kwargs):
        raise AssertionError("stream reached the RAG engine with a foreign session")

    monkeypatch.setattr(chat_mod, "ask_with_eval", fake_ask_with_eval)

    victim_session = _foreign_session("victim-user-2")
    _login_fresh_user(client)

    resp = client.post(
        "/api/v1/chat/stream",
        json={"question": "what was the account number?", "session_id": victim_session},
    )
    assert resp.status_code == 404, (
        f"expected 404 for a foreign session, got {resp.status_code}"
    )
    client.cookies.clear()


def test_chat_does_not_write_into_a_foreign_session(client, monkeypatch):
    """The rejection must happen before add_message persists anything."""
    import app.api.routes.chat as chat_mod
    from app.core.chat_store import get_recent_messages

    monkeypatch.setattr(chat_mod, "ask", lambda **kw: None)

    victim_session = _foreign_session("victim-user-3")
    before = len(get_recent_messages(victim_session, limit=50))

    _login_fresh_user(client)
    client.post(
        "/api/v1/chat/",
        json={"question": "injected", "session_id": victim_session},
    )

    after = get_recent_messages(victim_session, limit=50)
    assert len(after) == before, (
        "a message was written into another user's session: "
        f"{[m['content'] for m in after]}"
    )
    client.cookies.clear()


# ── SEC-7: CSP must not still permit the removed UI's scripts ────────────

def test_api_responses_deny_all_content_by_default(client):
    """A JSON-only service should allow nothing at all."""
    csp = client.get("/api/v1/health").headers["Content-Security-Policy"]
    assert "default-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp
    for stale in ("cdn.tailwindcss.com", "unpkg.com", "'unsafe-inline'"):
        assert stale not in csp, (
            f"CSP still carries {stale}, an allowance that existed only for "
            "the server-rendered UI that was removed"
        )


def test_docs_pages_keep_only_the_allowance_they_need(client):
    """Swagger/ReDoc need jsdelivr; that exception must not leak elsewhere."""
    csp = client.get("/docs").headers["Content-Security-Policy"]
    assert "cdn.jsdelivr.net" in csp
    assert "cdn.tailwindcss.com" not in csp
    assert "unpkg.com" not in csp


def test_xss_auditor_header_is_disabled(client):
    """'1; mode=block' is actively discouraged; CSP is the control."""
    assert client.get("/api/v1/health").headers["X-XSS-Protection"] == "0"


# ── SEC-8: every guard must be visible to dependency introspection ───────

def test_no_route_hand_rolls_its_auth_check():
    """A guard that is not a dependency is invisible to an authz audit.

    /auth/me originally re-derived the user from the cookie inline. It was
    correct, but an audit enumerating route dependencies reported it as
    unguarded, and it would not have inherited later hardening of
    require_authenticated_user.
    """
    from fastapi.routing import APIRoute

    from app.api.routes import admin, auth, chat, documents, feedback
    from app.core.auth import require_admin_user, require_authenticated_user

    guards = {require_authenticated_user, require_admin_user}
    # APIRoute.path already carries the router's prefix.
    intentionally_public = {"/auth/login", "/auth/register", "/auth/logout"}

    unguarded = []
    for router in (auth.router, chat.router, documents.router, feedback.router, admin.router):
        for r in router.routes:
            if not isinstance(r, APIRoute):
                continue
            if r.path in intentionally_public:
                continue
            found, stack = False, list(r.dependant.dependencies)
            while stack:
                d = stack.pop()
                if d.call in guards:
                    found = True
                    break
                stack.extend(d.dependencies)
            if not found:
                unguarded.append(f"{sorted(r.methods - {'HEAD'})} {r.path}")

    assert not unguarded, f"routes with no guard dependency: {unguarded}"
    # Guard the guard: if the traversal stops finding routes, this test would
    # pass vacuously the way the first version of the audit script did.
    assert sum(
        1 for rt in (auth.router, chat.router, documents.router, feedback.router, admin.router)
        for r in rt.routes if isinstance(r, APIRoute)
    ) >= 28, "route traversal found almost nothing - the check is not running"


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
