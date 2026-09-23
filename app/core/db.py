"""Shared SQLite setup for the relational stores.

Centralises three things the three stores (users, chats, evals) all need and
previously each did slightly differently:

* **Location** — databases live in ``settings.data_dir``, not inside the vector
  store directory, so the index can be rebuilt without touching user data.
* **Pragmas** — WAL for concurrent readers, and a ``busy_timeout`` so a
  concurrent writer waits for the lock instead of immediately raising
  ``database is locked``.
* **Migration** — a database left at the old ``vectorstore_dir`` location is
  moved on first open, WAL-checkpointed first so nothing is lost.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

#: How long a blocked writer waits for the lock before raising.
BUSY_TIMEOUT_MS = 5000


def resolve_db_path(db_name: str) -> Path:
    """Return the path for ``db_name``, migrating it out of the old location.

    Earlier versions stored these databases inside ``vectorstore_dir``. If a
    file is still there and the new location is empty, it is moved across.
    """
    settings = get_settings()
    new_path = Path(settings.data_dir) / db_name
    new_path.parent.mkdir(parents=True, exist_ok=True)

    if new_path.exists():
        return new_path

    legacy_path = Path(settings.vectorstore_dir) / db_name
    if not legacy_path.exists():
        return new_path

    # Fold any -wal content into the main file before moving, so we never
    # leave committed transactions behind in a stranded sidecar file.
    try:
        tmp = sqlite3.connect(str(legacy_path))
        try:
            tmp.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            tmp.commit()
        finally:
            tmp.close()
    except Exception:
        logger.exception("db_migration_checkpoint_failed", db=db_name)
        # Checkpoint failed — keep using the old file rather than risk a
        # partial move.
        return legacy_path

    try:
        legacy_path.replace(new_path)
        for suffix in ("-wal", "-shm"):
            sidecar = legacy_path.with_name(legacy_path.name + suffix)
            if sidecar.exists():
                sidecar.unlink()
        logger.info("db_migrated", db=db_name, frm=str(legacy_path), to=str(new_path))
    except Exception:
        logger.exception("db_migration_failed", db=db_name)
        return legacy_path

    return new_path


def connect(db_name: str) -> sqlite3.Connection:
    """Open a configured connection to ``db_name``."""
    path = resolve_db_path(db_name)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    logger.info("db_initialised", db=db_name, path=str(path))
    return conn


_local = threading.local()
_schema_ready: set[str] = set()
_schema_lock = threading.Lock()


def thread_connection(db_name: str, init: Callable[[sqlite3.Connection], None]) -> sqlite3.Connection:
    """This thread's connection to ``db_name``, opened on first use.

    A sqlite3 connection must not be used by two threads at once. The stores
    used to share one per process and lock only their writes, so overlapping
    requests - FastAPI runs sync dependencies such as the auth lookup in a
    thread pool - corrupted each other's cursors and failed with
    ``InterfaceError: bad parameter or other API misuse``. One connection per
    thread removes the sharing; WAL lets those connections read concurrently,
    and ``busy_timeout`` queues their writes.

    ``init`` creates and migrates the caller's tables. It runs once per process
    per store, not once per thread.
    """
    conns: dict[str, sqlite3.Connection] = getattr(_local, "conns", None) or {}
    _local.conns = conns
    conn = conns.get(db_name)
    if conn is None:
        conn = conns[db_name] = connect(db_name)
    key = f"{db_name}:{init.__module__}.{init.__qualname__}"
    if key not in _schema_ready:
        with _schema_lock:
            if key not in _schema_ready:
                init(conn)
                _schema_ready.add(key)
    return conn
