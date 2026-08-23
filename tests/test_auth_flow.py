"""Registration, login, lockout, and session-cookie behaviour."""

from __future__ import annotations

import uuid

import pytest


def test_register_then_login(client):
    username = f"user_{uuid.uuid4().hex[:10]}"
    password = "correct-horse-battery"

    resp = client.post("/register", data={"username": username, "password": password})
    assert resp.status_code == 200
    assert client.cookies.get("user_id"), "no session cookie set on register"

    client.cookies.clear()
    resp = client.post("/login", data={"username": username, "password": password})
    assert resp.status_code == 200
    assert client.cookies.get("user_id"), "no session cookie set on login"
    client.cookies.clear()


def test_login_with_wrong_password_is_rejected(client):
    username = f"user_{uuid.uuid4().hex[:10]}"
    client.post("/register", data={"username": username, "password": "correct-horse-battery"})
    client.cookies.clear()

    resp = client.post("/login", data={"username": username, "password": "wrong-password"})
    assert resp.status_code == 200          # form re-renders
    assert "Invalid username or password" in resp.text
    assert not client.cookies.get("user_id"), "session cookie set despite bad password"


# ── UX-1: the server minimum the UI must mirror ──────────────────────────

@pytest.mark.parametrize("password", ["", "short", "1234567"])
def test_passwords_under_eight_characters_rejected(password):
    from app.core.user_store import register_user

    with pytest.raises(ValueError, match="at least 8 characters"):
        register_user(f"user_{uuid.uuid4().hex[:8]}", password)


def test_login_form_advertises_the_real_minimum():
    """The UI told users 4 while the server required 8."""
    from pathlib import Path

    import app.main as main_mod

    root = Path(main_mod.__file__).resolve().parent.parent
    for name in ("login.html", "admin.html"):
        html = (root / "templates" / name).read_text(encoding="utf-8")
        assert "Min 4 characters" not in html, f"{name} still advertises a 4-char minimum"
    login = (root / "templates" / "login.html").read_text(encoding="utf-8")
    assert 'minlength="8"' in login


def test_duplicate_username_rejected():
    from app.core.user_store import register_user

    username = f"user_{uuid.uuid4().hex[:10]}"
    register_user(username, "correct-horse-battery")
    with pytest.raises(ValueError, match="already taken"):
        register_user(username, "another-good-password")


# ── Brute-force lockout ──────────────────────────────────────────────────

def test_lockout_after_repeated_failures():
    from app.core.auth import check_login_allowed, clear_failed_logins, record_failed_login

    username = f"user_{uuid.uuid4().hex[:10]}"
    assert check_login_allowed(username) is True
    for _ in range(5):
        record_failed_login(username)
    assert check_login_allowed(username) is False, "no lockout after 5 failures"

    clear_failed_logins(username)
    assert check_login_allowed(username) is True, "lockout not cleared on success"


# ── Session cookie signing ───────────────────────────────────────────────

def test_session_cookie_is_signed_and_tamper_evident():
    from app.core.auth import sign_user_id, unsign_user_id

    token = sign_user_id("user-123")
    assert token != "user-123", "cookie value is not signed"
    assert unsign_user_id(token) == "user-123"
    assert unsign_user_id(token[:-3] + "xyz") is None, "tampered token accepted"
    assert unsign_user_id("garbage") is None


# ── HYG-4: logout clears with matching attributes ────────────────────────

def test_logout_clears_the_session_cookie(client):
    username = f"user_{uuid.uuid4().hex[:10]}"
    client.post("/register", data={"username": username, "password": "correct-horse-battery"})
    assert client.cookies.get("user_id")

    resp = client.get("/logout", follow_redirects=False)
    assert resp.status_code == 302
    set_cookie = resp.headers.get("set-cookie", "")
    assert "user_id=" in set_cookie
    # Attributes must match those used when setting, or the browser keeps
    # the original cookie alongside the deletion.
    assert "Path=/" in set_cookie
    assert "SameSite=lax" in set_cookie or "samesite=lax" in set_cookie.lower()
    client.cookies.clear()


def test_protected_route_rejects_missing_cookie(client):
    client.cookies.clear()
    resp = client.get("/api/v1/documents/history")
    assert resp.status_code == 401
