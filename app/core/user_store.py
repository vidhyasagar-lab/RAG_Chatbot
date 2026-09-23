"""Lightweight SQLite store for user and document tracking."""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Generator

from app.core import db
from app.core.logging import get_logger

logger = get_logger(__name__)

_DB_NAME = "users.db"
_lock = Lock()


def _get_conn() -> sqlite3.Connection:
    """This thread's connection. Not one shared one: see db.thread_connection."""
    return db.thread_connection(_DB_NAME, _init_tables)


def _init_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id   TEXT PRIMARY KEY,
            username  TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS user_documents (
            doc_id      TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            filename    TEXT NOT NULL,
            file_size   INTEGER NOT NULL DEFAULT 0,
            chunks_added INTEGER NOT NULL DEFAULT 0,
            images_extracted INTEGER NOT NULL DEFAULT 0,
            status      TEXT NOT NULL DEFAULT 'ready',
            uploaded_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
        CREATE INDEX IF NOT EXISTS idx_user_documents_user
            ON user_documents(user_id);
    """)
    # Migrate: add password_hash column if missing (existing db without it)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
    if "password_hash" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT NOT NULL DEFAULT ''")
        conn.commit()
    if "role" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
        conn.commit()
    # The lifetime answer budget. A column rather than a COUNT over
    # chat_messages, because delete_session deletes a session's messages and a
    # derived count would refund the quota to anyone who cleared their history.
    # Existing accounts start at 0 rather than being charged for past chats.
    if "exchanges_used" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN exchanges_used INTEGER NOT NULL DEFAULT 0")
        conn.commit()


# ── User operations ──────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    """Hash a password with PBKDF2-SHA256."""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations=260_000)
    return salt.hex() + ":" + dk.hex()


def _verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its PBKDF2 hash."""
    try:
        salt_hex, dk_hex = password_hash.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations=260_000)
        # Constant-time: a plain == leaks how far the comparison matched.
        return hmac.compare_digest(dk.hex(), dk_hex)
    except (ValueError, AttributeError):
        return False


def register_user(username: str, password: str) -> dict[str, Any]:
    """Create a new user. Raises ValueError if username exists."""
    conn = _get_conn()
    username = username.strip()
    if not username:
        raise ValueError("Username cannot be empty")
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters")

    with _lock:
        existing = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,),
        ).fetchone()
        if existing:
            raise ValueError("Username already taken")

        user_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        pw_hash = _hash_password(password)
        conn.execute(
            "INSERT INTO users (user_id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, username, pw_hash, now),
        )
        conn.commit()
        logger.info("user_created", user_id=user_id, username=username)
        return {"user_id": user_id, "username": username, "created_at": now, "role": "user"}


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    """Verify credentials. Returns user dict or None."""
    conn = _get_conn()
    username = username.strip()
    row = conn.execute(
        "SELECT user_id, username, password_hash, created_at, COALESCE(role, 'user') as role, "
        "COALESCE(exchanges_used, 0) as exchanges_used FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    if not row:
        logger.warning("auth_user_not_found", username=username)
        return None
    if not _verify_password(password, row["password_hash"]):
        logger.warning("auth_bad_password", username=username, user_id=row["user_id"])
        return None
    return {"user_id": row["user_id"], "username": row["username"], "created_at": row["created_at"], "role": row["role"] if "role" in row.keys() else "user"}


def get_user(user_id: str) -> dict[str, Any] | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT user_id, username, created_at, COALESCE(role, 'user') as role, "
        "COALESCE(exchanges_used, 0) as exchanges_used FROM users WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    return dict(row) if row else None


def get_user_by_username(username: str) -> dict[str, Any] | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT user_id, username, created_at, COALESCE(role, 'user') as role, "
        "COALESCE(exchanges_used, 0) as exchanges_used FROM users WHERE username = ?",
        (username.strip(),),
    ).fetchone()
    return dict(row) if row else None


def increment_exchanges(user_id: str) -> int:
    """Charge one exchange against this user's lifetime budget.

    Returns the new total. Incremented in SQL rather than read-modify-write,
    so two concurrent answers cannot both read the same value and each store
    used+1, losing a charge.
    """
    conn = _get_conn()
    with _lock:
        conn.execute(
            "UPDATE users SET exchanges_used = COALESCE(exchanges_used, 0) + 1 WHERE user_id = ?",
            (user_id,),
        )
        conn.commit()
        row = conn.execute(
            "SELECT COALESCE(exchanges_used, 0) as used FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return row["used"] if row else 0


# ── Document operations ──────────────────────────────────────────────

def record_document(
    user_id: str,
    filename: str,
    file_size: int,
    chunks_added: int,
    images_extracted: int = 0,
) -> str:
    """Record a document upload for a user. Returns doc_id."""
    conn = _get_conn()
    doc_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()

    with _lock:
        conn.execute(
            """INSERT INTO user_documents
               (doc_id, user_id, filename, file_size, chunks_added, images_extracted, status, uploaded_at)
               VALUES (?, ?, ?, ?, ?, ?, 'ready', ?)""",
            (doc_id, user_id, filename, file_size, chunks_added, images_extracted, now),
        )
        conn.commit()

    logger.info("document_recorded", doc_id=doc_id, user_id=user_id, filename=filename)
    return doc_id


def get_user_documents(user_id: str) -> list[dict[str, Any]]:
    """Return all documents uploaded by a user, newest first."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT doc_id, filename, file_size, chunks_added, images_extracted,
                  status, uploaded_at
           FROM user_documents
           WHERE user_id = ?
           ORDER BY uploaded_at DESC""",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_user_stats(user_id: str) -> dict[str, Any]:
    """Return aggregate stats for a user."""
    conn = _get_conn()
    row = conn.execute(
        """SELECT COUNT(*) as total_docs,
                  COALESCE(SUM(chunks_added), 0) as total_chunks,
                  COALESCE(SUM(images_extracted), 0) as total_images
           FROM user_documents
           WHERE user_id = ?""",
        (user_id,),
    ).fetchone()
    return dict(row) if row else {"total_docs": 0, "total_chunks": 0, "total_images": 0}


def get_document(doc_id: str) -> dict[str, Any] | None:
    """Return a single document record by doc_id."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT doc_id, user_id, filename FROM user_documents WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    return dict(row) if row else None


def delete_document(doc_id: str) -> bool:
    """Delete a document record from user_documents. Returns True if a row was removed."""
    conn = _get_conn()
    with _lock:
        cur = conn.execute("DELETE FROM user_documents WHERE doc_id = ?", (doc_id,))
        conn.commit()
    return cur.rowcount > 0


# ── Admin operations ─────────────────────────────────────────────────

def list_all_users() -> list[dict[str, Any]]:
    """Return all users for admin view."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT u.user_id, u.username, COALESCE(u.role, 'user') as role, u.created_at,
                  COUNT(d.doc_id) as doc_count,
                  COALESCE(SUM(d.chunks_added), 0) as total_chunks
           FROM users u
           LEFT JOIN user_documents d ON u.user_id = d.user_id
           GROUP BY u.user_id
           ORDER BY u.created_at DESC""",
    ).fetchall()
    return [dict(r) for r in rows]


def admin_create_user(username: str, password: str, role: str = "user") -> dict[str, Any]:
    """Admin creates a user with specified role."""
    user = register_user(username, password)
    if role != "user":
        conn = _get_conn()
        with _lock:
            conn.execute("UPDATE users SET role = ? WHERE user_id = ?", (role, user["user_id"]))
            conn.commit()
        user["role"] = role
    return user


def admin_delete_user(user_id: str) -> bool:
    """Delete a user and all their documents."""
    conn = _get_conn()
    with _lock:
        conn.execute("DELETE FROM user_documents WHERE user_id = ?", (user_id,))
        cur = conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()
    return cur.rowcount > 0


def set_user_role(user_id: str, role: str) -> bool:
    """Update a user's role."""
    conn = _get_conn()
    with _lock:
        cur = conn.execute("UPDATE users SET role = ? WHERE user_id = ?", (role, user_id))
        conn.commit()
    return cur.rowcount > 0


def get_system_stats() -> dict[str, Any]:
    """Return system-wide stats for admin dashboard."""
    conn = _get_conn()
    users_row = conn.execute("SELECT COUNT(*) as total_users FROM users").fetchone()
    docs_row = conn.execute(
        """SELECT COUNT(*) as total_docs,
                  COALESCE(SUM(file_size), 0) as total_size,
                  COALESCE(SUM(chunks_added), 0) as total_chunks,
                  COALESCE(SUM(images_extracted), 0) as total_images
           FROM user_documents"""
    ).fetchone()
    return {
        "total_users": users_row["total_users"],
        "total_docs": docs_row["total_docs"],
        "total_size_bytes": docs_row["total_size"],
        "total_chunks": docs_row["total_chunks"],
        "total_images": docs_row["total_images"],
    }
