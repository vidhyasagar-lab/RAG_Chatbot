"""Idle timeout: a session ends an hour after its last authenticated request,
and every successful request restarts that hour."""

from __future__ import annotations

import time
import uuid

from itsdangerous import TimestampSigner, URLSafeTimedSerializer

from app.config import get_settings
from tests.helpers import sign_up

HOUR = 60 * 60


def _register(client) -> str:
    resp = sign_up(client, f"idle_{uuid.uuid4().hex[:10]}@example.test", "correct-horse-battery")
    assert resp.status_code == 201, resp.text
    client.cookies.clear()
    return resp.json()["user_id"]


def _token(user_id: str, age_seconds: int) -> str:
    """A session token as if it had been signed ``age_seconds`` ago."""

    class Past(TimestampSigner):
        def get_timestamp(self) -> int:
            return int(time.time()) - age_seconds

    return URLSafeTimedSerializer(get_settings().secret_key, salt="session-cookie", signer=Past).dumps(user_id)


def _session_cookies(resp) -> list[str]:
    return [c for c in resp.headers.get_list("set-cookie") if c.startswith("user_id=")]


def _me(client, token: str):
    return client.get("/api/v1/auth/me", headers={"cookie": f"user_id={token}"})


def test_login_cookie_lasts_the_idle_window_not_a_week(client):
    username = f"idle_{uuid.uuid4().hex[:10]}@example.test"
    resp = sign_up(client, username, "correct-horse-battery")
    client.cookies.clear()
    (cookie,) = _session_cookies(resp)
    assert f"Max-Age={get_settings().session_idle_minutes * 60}" in cookie


def test_session_idle_past_the_window_is_signed_out(client):
    uid = _register(client)
    resp = _me(client, _token(uid, HOUR + 5))
    assert resp.status_code == 401
    assert _session_cookies(resp) == [], "an expired session must not be revived"


def test_activity_restarts_the_window(client):
    uid = _register(client)
    resp = _me(client, _token(uid, 50 * 60))
    assert resp.status_code == 200
    (cookie,) = _session_cookies(resp)
    assert f"Max-Age={HOUR}" in cookie

    # The re-signed token counts from now: 50 more minutes idle is still fine.
    fresh = cookie.split(";", 1)[0].removeprefix("user_id=")
    from app.core.auth import unsign_user_id

    assert unsign_user_id(fresh) == uid


def test_a_just_signed_session_is_not_resent_on_every_response(client):
    uid = _register(client)
    resp = _me(client, _token(uid, 5))
    assert resp.status_code == 200
    assert _session_cookies(resp) == []


def test_logout_is_not_undone_by_the_refresh(client):
    uid = _register(client)
    resp = client.post("/api/v1/auth/logout", headers={"cookie": f"user_id={_token(uid, 30 * 60)}"})
    assert resp.status_code < 400
    cookies = _session_cookies(resp)
    assert len(cookies) == 1, cookies
    assert "Max-Age=0" in cookies[0] or 'user_id="";' in cookies[0] or "user_id=;" in cookies[0]


def test_tampered_token_is_not_refreshed(client):
    uid = _register(client)
    resp = _me(client, _token(uid, 30 * 60)[:-3] + "xyz")
    assert resp.status_code == 401
    assert _session_cookies(resp) == []
