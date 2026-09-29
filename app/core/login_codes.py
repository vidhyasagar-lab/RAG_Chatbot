"""One-time sign-in codes sent by email.

A six-digit code is only a million possibilities, which is nothing to a
script. What makes it safe is everything around it, so all of that lives
here rather than being spread across the route:

* **One live code per address.** Asking for a new one replaces the old, so
  an old mail in an inbox stops working the moment a newer one arrives.
* **A short life.** Ten minutes by default.
* **A hard attempt cap.** Five wrong guesses destroy the code entirely; the
  attacker has to ask for a new one, which alerts the owner by mail.
* **A resend cooldown**, so the request endpoint cannot be used to flood
  someone's inbox.
* **A daily ceiling** across the whole deployment, so a script cannot run up
  the provider's bill or get the sending address blacklisted.

Codes are stored as an HMAC under the app secret, never in the clear: a
leaked database then yields no usable codes, where a plain (or fast-hashed)
six-digit code would fall to an instant lookup.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import time
from datetime import date
from typing import Any

from app.config import get_settings
from app.core import db
from app.core.logging import get_logger

logger = get_logger(__name__)

_DB_NAME = "users.db"

#: Deliberately permissive. The authority on whether an address exists is
#: whether its owner can read the code we send; this only rejects input that
#: cannot be an address at all.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")

#: Addresses are compared lowercased, so Ada@x.com and ada@x.com are one
#: account rather than two. The local part is case-sensitive in the RFC and
#: case-insensitive at every provider anyone actually uses.
MAX_EMAIL_LENGTH = 254


class CodeRequestRefused(Exception):
    """A code was not issued. ``retry_after`` is in seconds, 0 if unknown."""

    def __init__(self, message: str, retry_after: int = 0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _init_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS login_codes (
            email      TEXT PRIMARY KEY,
            code_hash  TEXT NOT NULL,
            expires_at REAL NOT NULL,
            sent_at    REAL NOT NULL,
            attempts   INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS login_code_sends (
            day   TEXT PRIMARY KEY,
            count INTEGER NOT NULL DEFAULT 0
        );
    """)
    conn.commit()


def _conn() -> sqlite3.Connection:
    return db.thread_connection(_DB_NAME, _init_tables)


# ── Addresses ────────────────────────────────────────────────────────

def normalise_email(raw: str) -> str:
    """Trim and lowercase, so one address is one account."""
    return (raw or "").strip().lower()


def looks_like_email(value: str) -> bool:
    return bool(value) and len(value) <= MAX_EMAIL_LENGTH and bool(_EMAIL_RE.match(value))


# ── Codes ────────────────────────────────────────────────────────────

def _hash_code(email: str, code: str) -> str:
    """HMAC the code under the app secret, bound to the address.

    Binding to the address means a row lifted from the table cannot be
    replayed against a different account even if two people happen to hold
    the same code at the same moment.
    """
    secret = get_settings().secret_key.encode()
    return hmac.new(secret, f"{email}:{code}".encode(), hashlib.sha256).hexdigest()


def _generate_code() -> str:
    """A six-digit code from the system CSPRNG, leading zeros kept."""
    return f"{secrets.randbelow(1_000_000):06d}"


def _sends_today(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT count FROM login_code_sends WHERE day = ?", (date.today().isoformat(),)
    ).fetchone()
    return int(row["count"]) if row else 0


def _count_send(conn: sqlite3.Connection, delta: int) -> None:
    today = date.today().isoformat()
    conn.execute(
        "INSERT INTO login_code_sends (day, count) VALUES (?, MAX(0, ?)) "
        "ON CONFLICT(day) DO UPDATE SET count = MAX(0, count + ?)",
        (today, delta, delta),
    )
    conn.commit()


def issue_code(email: str) -> tuple[str, int]:
    """Create and store a code for ``email``. Returns (code, ttl_minutes).

    Raises CodeRequestRefused when another code was sent too recently, or
    when the deployment's daily ceiling has been reached. The count is
    incremented here rather than after a successful send, so a provider that
    accepts the message but is slow to report cannot be used to bypass the
    ceiling; ``discard_code`` gives the credit back when a send fails
    outright.
    """
    settings = get_settings()
    conn = _conn()
    now = time.time()

    existing = conn.execute(
        "SELECT sent_at FROM login_codes WHERE email = ?", (email,)
    ).fetchone()
    if existing:
        wait = settings.login_code_resend_seconds - (now - float(existing["sent_at"]))
        if wait > 0:
            raise CodeRequestRefused(
                "A code was just sent. Check your inbox before asking for another.",
                retry_after=max(1, int(wait)),
            )

    if _sends_today(conn) >= settings.login_code_daily_cap:
        logger.error("login_code_daily_cap_reached", cap=settings.login_code_daily_cap)
        raise CodeRequestRefused(
            "Too many sign-in codes have been sent today. Try again tomorrow, "
            "or sign in with your password."
        )

    code = _generate_code()
    ttl_minutes = max(1, settings.login_code_ttl_minutes)
    conn.execute(
        "INSERT INTO login_codes (email, code_hash, expires_at, sent_at, attempts) "
        "VALUES (?, ?, ?, ?, 0) "
        "ON CONFLICT(email) DO UPDATE SET "
        "  code_hash = excluded.code_hash, expires_at = excluded.expires_at, "
        "  sent_at = excluded.sent_at, attempts = 0",
        (email, _hash_code(email, code), now + ttl_minutes * 60, now),
    )
    conn.commit()
    _count_send(conn, 1)
    logger.info("login_code_issued", email=email, ttl_minutes=ttl_minutes)
    return code, ttl_minutes


def discard_code(email: str, refund: bool = False) -> None:
    """Drop a stored code. ``refund`` gives back its slot in the daily cap."""
    conn = _conn()
    conn.execute("DELETE FROM login_codes WHERE email = ?", (email,))
    conn.commit()
    if refund:
        _count_send(conn, -1)


def verify_code(email: str, code: str) -> bool:
    """Check a code and consume it. False for wrong, expired, or absent.

    A correct code is deleted on use, so it cannot be replayed. A wrong one
    counts against the attempt cap, and the last allowed failure deletes the
    code as well.
    """
    settings = get_settings()
    conn = _conn()
    row = conn.execute(
        "SELECT code_hash, expires_at, attempts FROM login_codes WHERE email = ?",
        (email,),
    ).fetchone()
    if not row:
        logger.warning("login_code_absent", email=email)
        return False

    if time.time() > float(row["expires_at"]):
        conn.execute("DELETE FROM login_codes WHERE email = ?", (email,))
        conn.commit()
        logger.warning("login_code_expired", email=email)
        return False

    # compare_digest, not ==: a plain comparison returns sooner on an earlier
    # mismatch, which leaks the code a character at a time under timing.
    if not hmac.compare_digest(row["code_hash"], _hash_code(email, (code or "").strip())):
        attempts = int(row["attempts"]) + 1
        if attempts >= settings.login_code_max_attempts:
            conn.execute("DELETE FROM login_codes WHERE email = ?", (email,))
            logger.warning("login_code_attempts_exhausted", email=email, attempts=attempts)
        else:
            conn.execute(
                "UPDATE login_codes SET attempts = ? WHERE email = ?", (attempts, email)
            )
            logger.warning("login_code_wrong", email=email, attempts=attempts)
        conn.commit()
        return False

    conn.execute("DELETE FROM login_codes WHERE email = ?", (email,))
    conn.commit()
    logger.info("login_code_accepted", email=email)
    return True


def attempts_remaining(email: str) -> int:
    """How many more wrong guesses this code survives. 0 when there is none."""
    conn = _conn()
    row = conn.execute("SELECT attempts FROM login_codes WHERE email = ?", (email,)).fetchone()
    if not row:
        return 0
    return max(0, get_settings().login_code_max_attempts - int(row["attempts"]))


def purge_expired() -> int:
    """Delete codes past their expiry. Returns how many went."""
    conn = _conn()
    cur = conn.execute("DELETE FROM login_codes WHERE expires_at < ?", (time.time(),))
    conn.commit()
    return cur.rowcount or 0


def debug_state(email: str) -> dict[str, Any] | None:
    """The stored row for tests and support, never the code itself."""
    row = _conn().execute(
        "SELECT email, expires_at, sent_at, attempts FROM login_codes WHERE email = ?",
        (email,),
    ).fetchone()
    return dict(row) if row else None
