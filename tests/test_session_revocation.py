"""Signing out must end the session, not just forget it locally.

The cookie signs a user id, so it was a bearer token with a 60-minute idle
life and nothing could retire it early. Logging out cleared the browser's
copy; anything that had already captured the value - a shared machine, a
proxy log, a backup - kept working until the window lapsed.

A version number now travels inside the signed payload and is checked
against the account. Bumping it retires every cookie issued before.
"""

from __future__ import annotations

import uuid

import pytest

from app.core import auth as auth_mod
from app.core import user_store
from tests.helpers import sign_up

PASSWORD = "correct-horse-battery"


@pytest.fixture
def signed_in(client):
    """A signed-in client, plus the cookie value it holds."""
    client.cookies.clear()
    username = f"revoke_{uuid.uuid4().hex[:8]}@example.test"
    resp = sign_up(client, username, PASSWORD)
    assert resp.status_code == 201, resp.text
    token = client.cookies.get("user_id")
    assert token
    yield client, username, token
    client.cookies.clear()


def _with_token(client, token: str):
    client.cookies.clear()
    client.cookies.set("user_id", token)
    return client


# ── The session still has to work ────────────────────────────────────

def test_a_signed_in_session_keeps_working(signed_in):
    client, _username, _token = signed_in

    assert client.get("/api/v1/auth/me").status_code == 200


def test_the_cookie_works_on_a_fresh_client(signed_in):
    """Ordinary behaviour: the cookie is what carries the session."""
    client, _username, token = signed_in

    assert _with_token(client, token).get("/api/v1/auth/me").status_code == 200


# ── Retiring it ──────────────────────────────────────────────────────

def test_a_cookie_captured_before_logout_stops_working(signed_in):
    """The finding. Clearing the browser's copy did nothing to this one."""
    client, _username, token = signed_in

    client.post("/api/v1/auth/logout")

    assert _with_token(client, token).get("/api/v1/auth/me").status_code == 401, (
        "a cookie captured before logout still authenticates"
    )


def test_changing_the_password_retires_existing_sessions(signed_in):
    """Changing a password is how someone responds to it being known. It
    has to end the sessions that knowledge may already have opened."""
    client, username, token = signed_in
    user = user_store.get_user_by_username(username)

    user_store.set_password(user["user_id"], "a-different-password-entirely")

    assert _with_token(client, token).get("/api/v1/auth/me").status_code == 401


def test_logging_out_does_not_retire_another_account(client):
    """The bump is per account, not global."""
    one = f"revoke_a_{uuid.uuid4().hex[:8]}@example.test"
    two = f"revoke_b_{uuid.uuid4().hex[:8]}@example.test"
    client.cookies.clear()
    sign_up(client, one, PASSWORD)
    token_one = client.cookies.get("user_id")
    client.cookies.clear()
    sign_up(client, two, PASSWORD)

    client.post("/api/v1/auth/logout")  # signs out two
    client.cookies.clear()

    assert _with_token(client, token_one).get("/api/v1/auth/me").status_code == 200


def test_signing_in_again_issues_a_working_cookie(signed_in):
    """Revocation must not leave the account unable to sign back in."""
    client, username, _token = signed_in
    client.post("/api/v1/auth/logout")
    client.cookies.clear()

    resp = client.post("/api/v1/auth/login", json={"username": username, "password": PASSWORD})

    assert resp.status_code == 200, resp.text
    assert client.get("/api/v1/auth/me").status_code == 200


# ── The token itself ─────────────────────────────────────────────────

def test_a_tampered_token_is_still_rejected(signed_in):
    client, _username, token = signed_in

    assert _with_token(client, token + "x").get("/api/v1/auth/me").status_code == 401


def test_a_token_from_before_versioning_is_not_accepted():
    """The old payload was a bare user id with no version to check. Those
    cannot be verified as current, so they are refused - everyone signs in
    once more after this ships, which is the cost of being able to revoke."""
    legacy = auth_mod._get_signer().dumps("some-user-id")

    assert auth_mod.unsign_user_id(legacy) is None


def test_a_token_naming_a_version_that_is_not_current_is_refused():
    username = f"stale_{uuid.uuid4().hex[:8]}@example.test"
    user = user_store.admin_create_user(username, PASSWORD)

    stale = auth_mod._get_signer().dumps([user["user_id"], 999])

    assert auth_mod.unsign_user_id(stale) is None
