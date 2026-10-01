"""Registration must prove the caller reads the address they signed up with.

``POST /auth/register`` created an account from nothing but a body: no code,
no invite, no verification. It is one of the handful of paths that bypass the
API key, so anyone who can reach the host could mint accounts, and each one
carries a lifetime answer budget and document slots that cost real money to
serve.

The machinery to fix it was already here and already used by the other door:
``/auth/code/verify`` has always created verified accounts. This closes the
one that did not.

Mail is never sent - conftest pins SMTP empty and codes are issued directly
through the store, the way tests/test_email_codes.py does it.
"""

from __future__ import annotations

import uuid

import pytest

from app.core import login_codes


def _address() -> str:
    return f"signup_{uuid.uuid4().hex[:10]}@example.test"


PASSWORD = "correct-horse-battery"


@pytest.fixture(autouse=True)
def _clean_cookies(client):
    client.cookies.clear()
    yield
    client.cookies.clear()


# ── The gate ─────────────────────────────────────────────────────────

def test_registering_without_a_code_is_refused(client):
    email = _address()

    resp = client.post("/api/v1/auth/register",
                       json={"username": email, "password": PASSWORD})

    assert resp.status_code == 422, (
        f"an account was created from nothing but a request body: {resp.status_code}"
    )


def test_no_account_is_left_behind_by_a_refused_registration(client):
    """The refusal must happen before the row is written, not after."""
    from app.core.user_store import get_user_by_username

    email = _address()
    client.post("/api/v1/auth/register", json={"username": email, "password": PASSWORD})

    assert get_user_by_username(email) is None


def test_registering_with_a_wrong_code_is_refused(client):
    email = _address()
    login_codes.issue_code(email)

    resp = client.post("/api/v1/auth/register",
                       json={"username": email, "password": PASSWORD, "code": "000000"})

    assert resp.status_code == 401


def test_a_code_issued_for_one_address_does_not_register_another(client):
    """Otherwise anyone with one working mailbox could mint any address."""
    mine, theirs = _address(), _address()
    code, _ = login_codes.issue_code(mine)

    resp = client.post("/api/v1/auth/register",
                       json={"username": theirs, "password": PASSWORD, "code": code})

    assert resp.status_code == 401


# ── The happy path ───────────────────────────────────────────────────

def test_a_verified_code_creates_the_account_with_its_password(client):
    email = _address()
    code, _ = login_codes.issue_code(email)

    resp = client.post("/api/v1/auth/register",
                       json={"username": email, "password": PASSWORD, "code": code})

    assert resp.status_code == 201, resp.text
    assert resp.json()["username"] == email


def test_registering_signs_the_person_in(client):
    """The session cookie was the point of the endpoint; keep it."""
    email = _address()
    code, _ = login_codes.issue_code(email)

    client.post("/api/v1/auth/register",
                json={"username": email, "password": PASSWORD, "code": code})

    assert client.get("/api/v1/auth/me").status_code == 200


def test_the_password_set_at_registration_works_afterwards(client):
    """The whole reason to keep a password step: the other door already
    creates accounts, but it leaves them with no password at all."""
    email = _address()
    code, _ = login_codes.issue_code(email)
    client.post("/api/v1/auth/register",
                json={"username": email, "password": PASSWORD, "code": code})
    client.cookies.clear()

    resp = client.post("/api/v1/auth/login", json={"username": email, "password": PASSWORD})

    assert resp.status_code == 200, resp.text


def test_the_code_is_spent_and_cannot_register_a_second_account(client):
    """A redeemed code must not be replayable."""
    first, second = _address(), _address()
    code, _ = login_codes.issue_code(first)
    client.post("/api/v1/auth/register",
                json={"username": first, "password": PASSWORD, "code": code})
    client.cookies.clear()

    resp = client.post("/api/v1/auth/register",
                       json={"username": second, "password": PASSWORD, "code": code})

    assert resp.status_code == 401


# ── Still an email address, still a real password ────────────────────

def test_a_valid_code_does_not_excuse_a_weak_password(client):
    email = _address()
    code, _ = login_codes.issue_code(email)

    resp = client.post("/api/v1/auth/register",
                       json={"username": email, "password": "short", "code": code})

    assert resp.status_code == 400
    assert "8" in resp.text, "the password floor stopped being reported"


def test_registering_an_address_that_already_has_an_account_is_refused(client):
    email = _address()
    code, _ = login_codes.issue_code(email)
    client.post("/api/v1/auth/register",
                json={"username": email, "password": PASSWORD, "code": code})
    client.cookies.clear()

    again, _ = login_codes.issue_code(email)
    resp = client.post("/api/v1/auth/register",
                       json={"username": email, "password": PASSWORD, "code": again})

    assert resp.status_code == 400


# ── The attempt cap still applies ────────────────────────────────────

def test_guessing_the_code_at_registration_burns_the_attempt_cap(client):
    """Registration must not be a way around the cap that protects the
    sign-in door, since both redeem the same code."""
    from app.config import get_settings

    email = _address()
    login_codes.issue_code(email)
    cap = get_settings().login_code_max_attempts

    for _ in range(cap):
        client.post("/api/v1/auth/register",
                    json={"username": email, "password": PASSWORD, "code": "000000"})

    assert login_codes.attempts_remaining(email) == 0, "the cap was not applied"
