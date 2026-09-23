"""SQLite store for golden dataset and evaluation results."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Any

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
        CREATE TABLE IF NOT EXISTS golden_dataset (
            id          TEXT PRIMARY KEY,
            question    TEXT NOT NULL,
            ground_truth TEXT NOT NULL,
            source_doc  TEXT NOT NULL DEFAULT '',
            source      TEXT NOT NULL DEFAULT 'synthetic',
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS eval_runs (
            id              TEXT PRIMARY KEY,
            run_type        TEXT NOT NULL DEFAULT 'batch',
            status          TEXT NOT NULL DEFAULT 'running',
            total_samples   INTEGER NOT NULL DEFAULT 0,
            completed       INTEGER NOT NULL DEFAULT 0,
            avg_faithfulness    REAL,
            avg_relevancy       REAL,
            avg_context_precision REAL,
            avg_context_recall  REAL,
            created_at      TEXT NOT NULL,
            completed_at    TEXT
        );

        CREATE TABLE IF NOT EXISTS eval_results (
            id              TEXT PRIMARY KEY,
            run_id          TEXT NOT NULL,
            golden_id       TEXT,
            question        TEXT NOT NULL,
            answer          TEXT NOT NULL DEFAULT '',
            contexts        TEXT NOT NULL DEFAULT '[]',
            ground_truth    TEXT NOT NULL DEFAULT '',
            faithfulness    REAL,
            answer_relevancy REAL,
            context_precision REAL,
            context_recall  REAL,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES eval_runs(id)
        );
        CREATE INDEX IF NOT EXISTS idx_eval_results_run
            ON eval_results(run_id);

        CREATE TABLE IF NOT EXISTS query_scores (
            trace_id    TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL DEFAULT '',
            faithfulness    REAL,
            answer_relevancy REAL,
            context_precision REAL,
            overall     REAL,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS eval_cache (
            cache_key   TEXT PRIMARY KEY,
            question    TEXT NOT NULL,
            faithfulness    REAL,
            context_precision REAL,
            answer_relevancy REAL,
            created_at  TEXT NOT NULL
        );
    """)
    # Migrate: query_scores gained user_id so score reads can be ownership
    # checked. Existing rows keep an empty owner and stay readable.
    cols = [r[1] for r in conn.execute("PRAGMA table_info(query_scores)").fetchall()]
    if "user_id" not in cols:
        conn.execute("ALTER TABLE query_scores ADD COLUMN user_id TEXT NOT NULL DEFAULT ''")
        conn.commit()


# ── Golden Dataset CRUD ──────────────────────────────────────────────

def add_golden_sample(question: str, ground_truth: str,
                      source_doc: str = "", source: str = "synthetic") -> dict:
    conn = _get_conn()
    sample_id = uuid.uuid4().hex[:16]
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn.execute(
            "INSERT INTO golden_dataset (id, question, ground_truth, source_doc, source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sample_id, question, ground_truth, source_doc, source, now),
        )
        conn.commit()
    return {"id": sample_id, "question": question, "ground_truth": ground_truth,
            "source_doc": source_doc, "source": source, "created_at": now}


def add_golden_samples_bulk(samples: list[dict]) -> int:
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for s in samples:
        rows.append((
            uuid.uuid4().hex[:16],
            s["question"],
            s["ground_truth"],
            s.get("source_doc", ""),
            s.get("source", "synthetic"),
            now,
        ))
    with _lock:
        conn.executemany(
            "INSERT INTO golden_dataset (id, question, ground_truth, source_doc, source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    return len(rows)


def get_golden_dataset() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM golden_dataset ORDER BY created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def delete_golden_sample(sample_id: str) -> bool:
    conn = _get_conn()
    with _lock:
        cur = conn.execute("DELETE FROM golden_dataset WHERE id = ?", (sample_id,))
        conn.commit()
    return cur.rowcount > 0


def clear_golden_dataset() -> int:
    conn = _get_conn()
    with _lock:
        cur = conn.execute("DELETE FROM golden_dataset")
        conn.commit()
    return cur.rowcount


# ── Eval Runs ────────────────────────────────────────────────────────

def create_eval_run(run_type: str = "batch", total_samples: int = 0) -> str:
    conn = _get_conn()
    run_id = uuid.uuid4().hex[:16]
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn.execute(
            "INSERT INTO eval_runs (id, run_type, status, total_samples, created_at) "
            "VALUES (?, ?, 'running', ?, ?)",
            (run_id, run_type, total_samples, now),
        )
        conn.commit()
    return run_id


def save_eval_result(run_id: str, question: str, answer: str,
                     contexts: list[str], ground_truth: str,
                     scores: dict, golden_id: str | None = None) -> None:
    conn = _get_conn()
    result_id = uuid.uuid4().hex[:16]
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn.execute(
            "INSERT INTO eval_results "
            "(id, run_id, golden_id, question, answer, contexts, ground_truth, "
            "faithfulness, answer_relevancy, context_precision, context_recall, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                result_id, run_id, golden_id, question, answer,
                json.dumps(contexts), ground_truth,
                scores.get("faithfulness"),
                scores.get("answer_relevancy"),
                scores.get("context_precision"),
                scores.get("context_recall"),
                now,
            ),
        )
        # Update completed count
        conn.execute(
            "UPDATE eval_runs SET completed = completed + 1 WHERE id = ?",
            (run_id,),
        )
        conn.commit()


def complete_eval_run(run_id: str) -> None:
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        # Compute averages from results
        row = conn.execute(
            "SELECT AVG(faithfulness) as f, AVG(answer_relevancy) as r, "
            "AVG(context_precision) as cp, AVG(context_recall) as cr "
            "FROM eval_results WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        conn.execute(
            "UPDATE eval_runs SET status = 'completed', completed_at = ?, "
            "avg_faithfulness = ?, avg_relevancy = ?, "
            "avg_context_precision = ?, avg_context_recall = ? "
            "WHERE id = ?",
            (now, row["f"], row["r"], row["cp"], row["cr"], run_id),
        )
        conn.commit()


def fail_eval_run(run_id: str, error: str = "") -> None:
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn.execute(
            "UPDATE eval_runs SET status = 'failed', completed_at = ? WHERE id = ?",
            (now, run_id),
        )
        conn.commit()


def get_eval_runs() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT 20"
    ).fetchall()
    return [dict(r) for r in rows]


def get_eval_run(run_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def get_eval_results(run_id: str) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM eval_results WHERE run_id = ? ORDER BY created_at",
        (run_id,),
    ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        d["contexts"] = json.loads(d["contexts"])
        results.append(d)
    return results


# ── Per-Query Scores ─────────────────────────────────────────────────

def save_query_scores(trace_id: str, scores: dict, user_id: str = "") -> None:
    """Store per-query RAGAS scores keyed by trace_id.

    ``user_id`` records who the trace belongs to so score reads and feedback
    writes can be restricted to their owner.
    """
    conn = _get_conn()
    vals = [scores.get(k) for k in ("faithfulness", "answer_relevancy", "context_precision")]
    valid = [v for v in vals if v is not None]
    overall = sum(valid) / len(valid) if valid else None
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn.execute(
            "INSERT OR REPLACE INTO query_scores "
            "(trace_id, user_id, faithfulness, answer_relevancy, context_precision, overall, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (trace_id, user_id, *vals, overall, now),
        )
        conn.commit()


def get_query_scores(trace_id: str, user_id: str | None = None) -> dict | None:
    """Retrieve per-query RAGAS scores by trace_id.

    When ``user_id`` is given, a trace owned by someone else reads as absent
    rather than raising — the caller cannot distinguish "not yours" from
    "not ready yet", which avoids leaking which trace ids exist.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM query_scores WHERE trace_id = ?", (trace_id,)
    ).fetchone()
    if not row:
        return None
    record = dict(row)
    if user_id is not None:
        owner = record.get("user_id") or ""
        # Rows predating the user_id column have an empty owner; treat those
        # as unclaimed rather than locking everyone out of old traces.
        if owner and owner != user_id:
            return None
    return record


def trace_belongs_to(trace_id: str, user_id: str) -> bool:
    """True if ``trace_id`` is unowned or owned by ``user_id``."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT user_id FROM query_scores WHERE trace_id = ?", (trace_id,)
    ).fetchone()
    if not row:
        # No score row yet — evaluation may still be running, so we cannot
        # prove ownership either way. Allow it; the trace id is unguessable.
        return True
    owner = row["user_id"] or ""
    return not owner or owner == user_id


# ── Eval Cache (keyed by question + context + answer hash) ───────────

def _make_cache_key(question: str, contexts: list[str], answer: str = "") -> str:
    """Create a deterministic hash from question + contexts + answer.

    The answer belongs in the key because faithfulness measures the answer
    against the contexts. Keying on question + contexts alone returns one
    answer's score for a different answer to the same question, which is how
    an ungrounded reply could inherit a passing verdict.

    ``answer=""`` is still accepted, for a caller that wants a row keyed on the
    question and contexts alone. No production caller does: context_precision
    looked answer-independent but is not — ragas judges each context against
    the response — so it shares this answer-keyed row.
    """
    import hashlib
    normalised = (
        question.strip().lower()
        + "||" + "||".join(c.strip() for c in sorted(contexts))
        + "||" + answer.strip()
    )
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:32]


def get_eval_cache(question: str, contexts: list[str], answer: str = "") -> dict | None:
    """Look up cached eval scores for this question + context + answer combo."""
    conn = _get_conn()
    key = _make_cache_key(question, contexts, answer)
    row = conn.execute(
        "SELECT faithfulness, context_precision, answer_relevancy FROM eval_cache WHERE cache_key = ?",
        (key,),
    ).fetchone()
    if row:
        logger.info("eval_cache_hit", cache_key=key[:8])
        return dict(row)
    return None


def save_eval_cache(
    question: str, contexts: list[str], scores: dict, answer: str = ""
) -> None:
    """Save eval scores to cache keyed by question + contexts + answer."""
    conn = _get_conn()
    key = _make_cache_key(question, contexts, answer)
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn.execute(
            "INSERT OR REPLACE INTO eval_cache "
            "(cache_key, question, faithfulness, context_precision, answer_relevancy, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                key,
                question.strip(),
                scores.get("faithfulness"),
                scores.get("context_precision"),
                scores.get("answer_relevancy"),
                now,
            ),
        )
        conn.commit()
    logger.info("eval_cache_saved", cache_key=key[:8])
