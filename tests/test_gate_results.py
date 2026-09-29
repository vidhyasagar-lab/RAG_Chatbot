"""The gate's verdict, and any revised answer, reach the reader after the stream.

The gate runs after `done`, so its result is stored under the trace id and
fetched from /chat/gate/{trace_id}; the stored chat message is updated too, so
a chat reopened from history shows the answer that survived the check.
"""

from __future__ import annotations

import json
import uuid

import pytest

from app.core.chat_store import add_message, create_session, get_session_messages, revise_answer_message
from app.core.eval_store import get_gate_result, save_gate_result

# faithfulness is the rewrite's; draft_faithfulness the rejected draft's.
RESULT = {"verdict": "rejected", "faithfulness": 0.86, "context_precision": 0.8,
          "threshold": 0.5, "revised_answer": "the grounded answer",
          "draft_faithfulness": 0.2}


def _trace() -> str:
    return uuid.uuid4().hex


# ── store ────────────────────────────────────────────────────────────

def test_a_saved_result_reads_back_for_its_owner():
    trace = _trace()
    save_gate_result(trace, "alice", RESULT)

    assert get_gate_result(trace, "alice") == RESULT


def test_a_table_from_before_draft_scores_gains_the_column():
    """The VM's gate_results predates draft_faithfulness."""
    import sqlite3

    from app.core import eval_store

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE gate_results (trace_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, "
                 "verdict TEXT NOT NULL, faithfulness REAL, context_precision REAL, "
                 "threshold REAL NOT NULL, revised_answer TEXT, created_at TEXT NOT NULL)")
    eval_store._init_tables(conn)

    cols = [r[1] for r in conn.execute("PRAGMA table_info(gate_results)").fetchall()]
    assert "draft_faithfulness" in cols


def test_someone_elses_result_reads_as_absent():
    trace = _trace()
    save_gate_result(trace, "alice", RESULT)

    assert get_gate_result(trace, "mallory") is None


def test_a_pending_result_reads_as_absent():
    assert get_gate_result(_trace(), "alice") is None


# ── the stored message follows the verdict ───────────────────────────

def _answer_with_trace(trace: str) -> str:
    session = create_session("alice", "t")
    add_message(session["session_id"], "user", "q")
    add_message(session["session_id"], "assistant", "the draft",
                meta={"trace_id": trace, "sources": []})
    return session["session_id"]


def test_a_revised_answer_replaces_the_stored_draft():
    trace = _trace()
    session = _answer_with_trace(trace)

    revise_answer_message(trace, "the grounded answer", {"verdict": "rejected", "attempt": 2})

    answer = get_session_messages(session)[-1]
    assert answer["content"] == "the grounded answer"
    assert answer["meta"]["eval"] == {"verdict": "rejected", "attempt": 2}
    assert answer["meta"]["trace_id"] == trace


def test_a_passing_verdict_keeps_the_text_and_records_the_verdict():
    trace = _trace()
    session = _answer_with_trace(trace)

    revise_answer_message(trace, None, {"verdict": "passed", "attempt": 1})

    answer = get_session_messages(session)[-1]
    assert answer["content"] == "the draft"
    assert answer["meta"]["eval"]["verdict"] == "passed"


def test_an_unknown_trace_changes_nothing():
    trace = _trace()
    session = _answer_with_trace(trace)

    revise_answer_message(_trace(), "not this one", {"verdict": "rejected"})

    assert get_session_messages(session)[-1]["content"] == "the draft"


# ── endpoint ─────────────────────────────────────────────────────────

@pytest.fixture
def signed_in(client):
    client.cookies.clear()
    name = f"gate_{uuid.uuid4().hex[:10]}@example.test"
    resp = client.post("/api/v1/auth/register",
                       json={"username": name, "password": "correct-horse-battery"})
    assert resp.status_code == 201
    yield client, client.get("/api/v1/auth/me").json()["user_id"]
    client.cookies.clear()


def test_the_endpoint_returns_204_until_the_gate_finishes(signed_in):
    client, _ = signed_in
    assert client.get(f"/api/v1/chat/gate/{_trace()}").status_code == 204


def test_the_endpoint_returns_the_owners_result(signed_in):
    client, user_id = signed_in
    trace = _trace()
    save_gate_result(trace, user_id, RESULT)

    resp = client.get(f"/api/v1/chat/gate/{trace}")

    assert resp.status_code == 200
    assert resp.json() == RESULT


def test_a_gate_that_finishes_before_the_answer_is_stored_is_not_lost(signed_in, monkeypatch):
    """The gate can beat the route to it - an instant metric failure, say.
    The route then finds no message to update, so it must pick the result up
    itself when it stores the answer."""
    import app.api.routes.chat as chat_routes

    client, user_id = signed_in
    trace = _trace()

    def instant_gate(question, chat_history=None, top_k=None, user_id="", session_id="", gated=None):
        async def gen():
            yield "data: " + json.dumps({"type": "meta", "sources": [], "images": [],
                                         "trace_id": trace, "session_id": session_id}) + "\n\n"
            yield "data: " + json.dumps({"type": "token", "content": "the draft", "attempt": 1}) + "\n\n"
            save_gate_result(trace, user_id, RESULT)   # finished before `done` was handled
            yield "data: " + json.dumps({"type": "done", "final_attempt": 1, "gate": "pending"}) + "\n\n"
        return gen()

    monkeypatch.setattr(chat_routes, "ask_stream", instant_gate)
    resp = client.post("/api/v1/chat/stream", json={"question": "q"})
    session_id = json.loads(next(l[6:] for l in resp.text.splitlines() if l.startswith("data: ")))["session_id"]

    answer = client.get(f"/api/v1/chat/sessions/{session_id}").json()["messages"][-1]
    assert answer["content"] == "the grounded answer"
    assert answer["eval"]["verdict"] == "rejected"
    # The rewrite's score, not the rejected draft's.
    assert answer["eval"]["faithfulness"] == 0.86
    assert answer["eval"]["draft_faithfulness"] == 0.2


def test_the_endpoint_hides_other_users_results(signed_in):
    client, _ = signed_in
    trace = _trace()
    save_gate_result(trace, "someone-else", RESULT)

    assert client.get(f"/api/v1/chat/gate/{trace}").status_code == 204
