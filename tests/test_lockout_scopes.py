"""One door's failures must not close the other.

The lockout is keyed by identity, and the identity is an address, so the
password door and the code door shared a single counter: five wrong password
guesses locked the person out of signing in by emailed code as well - the
very door they would reach for when they could not remember the password.
Worse, an attacker who knows an address could close both doors on demand
without knowing anything else.

The counters are namespaced instead. Deliberately not keyed per client
address: a distributed attacker would then get five guesses per address
rather than five in total, which is the attack the lockout exists to stop.
A five-minute wait with the other door open is the better trade.
"""

from __future__ import annotations

import uuid

import pytest

from app.config import get_settings
from app.core import auth as auth_mod
from app.core import login_codes
from tests.helpers import sign_up

PASSWORD = "correct-horse-battery"


@pytest.fixture(autouse=True)
def _clean(client):
    client.cookies.clear()
    yield
    client.cookies.clear()


def _burn_password_attempts(client, address, times=5):
    for _ in range(times):
        client.post("/api/v1/auth/login",
                    json={"username": address, "password": "definitely-wrong"})


def _burn_code_attempts(client, address, times=5):
    for _ in range(times):
        client.post("/api/v1/auth/code/verify",
                    json={"email": address, "code": "000000"})


def test_wrong_passwords_do_not_close_the_code_door(client):
    address = f"lock_{uuid.uuid4().hex[:8]}@example.test"
    sign_up(client, address, PASSWORD)
    client.cookies.clear()

    _burn_password_attempts(client, address)

    # The code door should still issue and accept a code.
    code, _ = login_codes.issue_code(login_codes.normalise_email(address))
    resp = client.post("/api/v1/auth/code/verify", json={"email": address, "code": code})

    assert resp.status_code == 200, (
        f"the password door's failures locked the code door: {resp.status_code} {resp.text}"
    )


def test_wrong_codes_do_not_close_the_password_door(client):
    address = f"lock_{uuid.uuid4().hex[:8]}@example.test"
    sign_up(client, address, PASSWORD)
    client.cookies.clear()

    _burn_code_attempts(client, address)

    resp = client.post("/api/v1/auth/login", json={"username": address, "password": PASSWORD})

    assert resp.status_code == 200, (
        f"the code door's failures locked the password door: {resp.status_code} {resp.text}"
    )


# ── Each door still locks ────────────────────────────────────────────

def test_the_password_door_still_locks_after_enough_wrong_guesses(client):
    address = f"lock_{uuid.uuid4().hex[:8]}@example.test"
    sign_up(client, address, PASSWORD)
    client.cookies.clear()

    _burn_password_attempts(client, address)

    # Even the right password is refused while locked.
    resp = client.post("/api/v1/auth/login", json={"username": address, "password": PASSWORD})

    assert resp.status_code == 429
    assert resp.headers.get("retry-after"), "the client is not told when to try again"


def test_the_code_door_still_locks_after_enough_wrong_guesses(client):
    address = f"lock_{uuid.uuid4().hex[:8]}@example.test"
    _burn_code_attempts(client, address, times=get_settings().login_code_max_attempts + 1)

    code, _ = login_codes.issue_code(login_codes.normalise_email(address))
    resp = client.post("/api/v1/auth/code/verify", json={"email": address, "code": code})

    assert resp.status_code == 429


def test_the_scopes_are_separate_keys():
    """Asserted directly too, so the reason survives a refactor."""
    address = f"lock_{uuid.uuid4().hex[:8]}@example.test"

    for _ in range(5):
        auth_mod.record_failed_login(address, scope="password")

    assert auth_mod.check_login_allowed(address, scope="password") is False
    assert auth_mod.check_login_allowed(address, scope="code") is True


def test_clearing_one_scope_leaves_the_other(client):
    address = f"lock_{uuid.uuid4().hex[:8]}@example.test"
    auth_mod.record_failed_login(address, scope="code")
    auth_mod.record_failed_login(address, scope="password")

    auth_mod.clear_failed_logins(address, scope="password")

    # The code scope keeps its single recorded failure; nothing is locked yet,
    # but the counters must not have been merged.
    assert auth_mod.check_login_allowed(address, scope="code") is True
    for _ in range(4):
        auth_mod.record_failed_login(address, scope="code")
    assert auth_mod.check_login_allowed(address, scope="code") is False
    assert auth_mod.check_login_allowed(address, scope="password") is True
