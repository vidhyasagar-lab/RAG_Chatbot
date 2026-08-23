"""SQLite store for chat sessions and messages."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from app.core import db
from app.core.logging import get_logger

logger = get_logger(__name__)

_DB_NAME = "users.db"  # reuse existing DB
_lock = Lock()
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    """Return the shared connection, opening it on first use."""
    global _conn
    if _conn is not None:
        return _conn
    with _lock:
        if _conn is not None:
            return _conn
        # db.connect() applies WAL + busy_timeout and migrates the file
        # out of the old vectorstore_dir location if it is still there.
        _conn = db.connect(_DB_NAME)
        _init_tables(_conn)
        return _conn


def _init_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id  TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            title       TEXT NOT NULL DEFAULT 'New Chat',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chat_sessions_user
            ON chat_sessions(user_id, updated_at DESC);

        CREATE TABLE IF NOT EXISTS chat_messages (
            message_id  TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
                ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_chat_messages_session
            ON chat_messages(session_id, created_at);
    """)


# ── Session operations ───────────────────────────────────────────────

def create_session(user_id: str, title: str = "New Chat") -> dict[str, Any]:
    conn = _get_conn()
    session_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn.execute(
            "INSERT INTO chat_sessions (session_id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, title, now, now),
        )
        conn.commit()
    return {"session_id": session_id, "user_id": user_id, "title": title,
            "created_at": now, "updated_at": now}


def get_user_sessions(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        """SELECT session_id, title, created_at, updated_at
           FROM chat_sessions WHERE user_id = ?
           ORDER BY updated_at DESC LIMIT ?""",
        (user_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> dict[str, Any] | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT session_id, user_id, title, created_at, updated_at FROM chat_sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return dict(row) if row else None


def update_session_title(session_id: str, title: str) -> None:
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE session_id = ?",
            (title, now, session_id),
        )
        conn.commit()


def delete_session(session_id: str) -> None:
    conn = _get_conn()
    with _lock:
        conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        conn.commit()


def _touch_session(conn: sqlite3.Connection, session_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
        (now, session_id),
    )


# ── Message operations ───────────────────────────────────────────────

def add_message(session_id: str, role: str, content: str) -> str:
    conn = _get_conn()
    message_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn.execute(
            "INSERT INTO chat_messages (message_id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (message_id, session_id, role, content, now),
        )
        _touch_session(conn, session_id)
        conn.commit()
    return message_id


def get_session_messages(session_id: str) -> list[dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        """SELECT message_id, role, content, created_at
           FROM chat_messages WHERE session_id = ?
           ORDER BY created_at ASC""",
        (session_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_recent_messages(session_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """Return the last N messages for building chat context."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT role, content FROM chat_messages
           WHERE session_id = ?
           ORDER BY created_at DESC LIMIT ?""",
        (session_id, limit),
    ).fetchall()
    return [dict(r) for r in reversed(rows)]
