"""How passwords are hashed, and what the hashing gives away.

Two findings from the audit live here. Both are about work that must happen
whether or not it is needed.
"""

from __future__ import annotations

import hashlib
import uuid

import pytest

from app.core import user_store
from tests.helpers import sign_up


@pytest.fixture
def count_kdf(monkeypatch):
    """Count the key-derivation calls, which is where the time goes."""
    calls: list[tuple] = []
    real = hashlib.pbkdf2_hmac

    def counting(*args, **kwargs):
        calls.append(args[:2])
        return real(*args, **kwargs)

    monkeypatch.setattr(user_store.hashlib, "pbkdf2_hmac", counting)
    return calls


# ── A username oracle in the timing ──────────────────────────────────
#
# Login deliberately returns one message for "no such user" and "wrong
# password" so the endpoint cannot be used to find out who has an account.
# The work behind it gave the answer away anyway: an unknown address returned
# before any key derivation, while a real one spent 600,000 rounds of it. The
# difference is tens of milliseconds and perfectly measurable.
#
# Asserted as "the work happened", not as elapsed time, because a timing
# assertion on a shared test runner is a coin flip.

def test_an_unknown_user_still_costs_a_key_derivation(count_kdf):
    user_store.authenticate_user(f"ghost_{uuid.uuid4().hex[:8]}@example.test", "whatever")

    assert len(count_kdf) == 1, (
        "no key derivation ran for an unknown address, so the endpoint answers "
        "faster for addresses that do not exist - a username oracle in the timing"
    )


def test_a_real_user_costs_the_same_one_derivation(client, count_kdf):
    username = f"timing_{uuid.uuid4().hex[:8]}@example.test"
    sign_up(client, username, "correct-horse-battery")
    count_kdf.clear()

    user_store.authenticate_user(username, "wrong-password-entirely")

    assert len(count_kdf) == 1


def test_an_account_with_no_password_still_costs_a_derivation(client, count_kdf):
    """Accounts created by the code door have an empty hash. Failing fast on
    those would say which accounts have never set a password."""
    username = f"nopw_{uuid.uuid4().hex[:8]}@example.test"
    user_store.get_or_create_user_by_email(username)
    count_kdf.clear()

    assert user_store.authenticate_user(username, "anything") is None
    assert len(count_kdf) == 1


def test_the_dummy_verification_cannot_be_logged_into():
    """The stand-in hash must not be a hash of anything guessable."""
    assert user_store._verify_password("", user_store._DUMMY_HASH) is False
    assert user_store._verify_password("password", user_store._DUMMY_HASH) is False
    assert user_store._verify_password(user_store._DUMMY_HASH, user_store._DUMMY_HASH) is False


# ── Iteration count, and moving it without a reset ───────────────────

def test_new_passwords_use_the_current_iteration_count():
    stored = user_store._hash_password("correct-horse-battery")

    assert stored.startswith(f"pbkdf2${user_store.PBKDF2_ROUNDS}$")
    assert user_store.PBKDF2_ROUNDS >= 600_000, (
        "600,000 is the current OWASP figure for PBKDF2-HMAC-SHA256"
    )


def test_a_password_stored_the_old_way_still_verifies():
    """The old format was salt_hex:dk_hex at 260,000 rounds, with the count
    recorded nowhere. Rejecting it would lock out every existing account."""
    import os

    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", b"correct-horse-battery", salt, 260_000)
    legacy = salt.hex() + ":" + dk.hex()

    assert user_store._verify_password("correct-horse-battery", legacy) is True
    assert user_store._verify_password("wrong", legacy) is False


def test_signing_in_upgrades_a_password_stored_the_old_way(client):
    """Nobody is asked to reset anything: the hash is rewritten at the next
    successful sign-in, when the plaintext is in hand anyway."""
    import os

    username = f"legacy_{uuid.uuid4().hex[:8]}@example.test"
    password = "correct-horse-battery"
    sign_up(client, username, password)

    # Write the old format directly, as an account created before the change
    # would have it.
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
    conn = user_store._get_conn()
    conn.execute("UPDATE users SET password_hash = ? WHERE username = ?",
                 (salt.hex() + ":" + dk.hex(), username))
    conn.commit()

    assert user_store.authenticate_user(username, password) is not None

    row = conn.execute("SELECT password_hash FROM users WHERE username = ?",
                       (username,)).fetchone()
    assert row["password_hash"].startswith("pbkdf2$"), "the hash was not upgraded"
    # And the upgraded hash still accepts the same password.
    assert user_store.authenticate_user(username, password) is not None


def test_a_failed_sign_in_does_not_rewrite_the_hash(client):
    """Only a correct password proves what to re-hash."""
    import os

    username = f"nochange_{uuid.uuid4().hex[:8]}@example.test"
    sign_up(client, username, "correct-horse-battery")
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", b"correct-horse-battery", salt, 260_000)
    legacy = salt.hex() + ":" + dk.hex()
    conn = user_store._get_conn()
    conn.execute("UPDATE users SET password_hash = ? WHERE username = ?", (legacy, username))
    conn.commit()

    user_store.authenticate_user(username, "the-wrong-one")

    row = conn.execute("SELECT password_hash FROM users WHERE username = ?",
                       (username,)).fetchone()
    assert row["password_hash"] == legacy
