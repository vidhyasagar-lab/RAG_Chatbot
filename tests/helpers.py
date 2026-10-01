"""Shared test helpers."""

from __future__ import annotations


def sign_up(client, username: str, password: str = "correct-horse-battery"):
    """Register through the API, proving the address the way a person does.

    Registration requires an emailed code (tests/test_verified_signup.py), so
    a bare POST to /auth/register is now a 422. No mail is sent here: the code
    is issued straight from the store, which is the same value the endpoint
    would have mailed.

    Every test that just needs an account goes through this, which leaves
    test_verified_signup.py as the only place the raw request shape is written
    out - so the gate has exactly one description, and tests that merely need
    a signed-in user do not quietly become tests of registration.
    """
    from app.core.login_codes import issue_code, normalise_email

    code, _ttl = issue_code(normalise_email(username))
    return client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": password, "code": code},
    )
