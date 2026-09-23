"""Per-user quotas: 25 exchanges for the life of the account, 2 documents held.

Each answer costs real Azure spend on a 2 GB box open to the internet, so a
registered account gets a finite budget. The exchange cap is a lifetime one -
it never resets - which is why it cannot be derived by counting rows in
chat_messages: delete_session removes a session's messages, so anyone could
clear their history and start over. It lives in its own column instead.

The document cap is the opposite: it counts what a user currently holds, so
deleting a document frees the slot. That caps disk and index size rather than
ingest spend, which is the trade the design chose deliberately.

Admins are exempt from both.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

from app.config import get_settings
from app.core import quota


# ── the policy, without the web layer ────────────────────────────────

def _user(role: str = "user", used: int = 0) -> dict:
    return {"user_id": "u1", "username": "someone", "role": role, "exchanges_used": used}


def test_a_user_under_the_limit_may_ask():
    quota.check_exchange_quota(_user(used=24))  # does not raise


def test_a_user_at_the_limit_may_not_ask():
    with pytest.raises(HTTPException) as exc:
        quota.check_exchange_quota(_user(used=25))
    assert exc.value.status_code == 403
    assert "25" in exc.value.detail


def test_the_exchange_limit_is_403_not_429():
    """429 already means two other things here: the rate limiter and the login
    lockout. Both clear on their own; a lifetime quota never does."""
    with pytest.raises(HTTPException) as exc:
        quota.check_exchange_quota(_user(used=99))
    assert exc.value.status_code == 403


def test_an_admin_is_never_out_of_exchanges():
    quota.check_exchange_quota(_user(role="admin", used=10_000))


def test_a_user_below_the_document_cap_may_upload():
    quota.check_document_quota(_user(), held=1)


def test_a_user_at_the_document_cap_may_not_upload():
    with pytest.raises(HTTPException) as exc:
        quota.check_document_quota(_user(), held=2)
    assert exc.value.status_code == 403


def test_an_admin_is_never_out_of_document_slots():
    quota.check_document_quota(_user(role="admin"), held=500)


def test_limits_come_from_settings():
    s = get_settings()
    assert s.max_exchanges_per_user == 25
    assert s.max_documents_per_user == 2


def test_usage_reports_both_budgets():
    snapshot = quota.usage_for(_user(used=7), held=1)
    assert snapshot == {
        "exchanges_used": 7,
        "exchanges_limit": 25,
        "documents_used": 1,
        "documents_limit": 2,
    }


def test_an_admin_reports_no_limit():
    """The UI needs to tell 'plenty left' from 'unlimited'."""
    snapshot = quota.usage_for(_user(role="admin", used=7), held=1)
    assert snapshot["exchanges_limit"] is None
    assert snapshot["documents_limit"] is None


# ── the counter, which must survive history deletion ─────────────────

def test_the_counter_starts_at_zero_and_increments():
    from app.core.user_store import get_user, increment_exchanges, register_user

    u = register_user(f"q_{uuid.uuid4().hex[:8]}", "correct-horse-battery")
    assert get_user(u["user_id"])["exchanges_used"] == 0

    increment_exchanges(u["user_id"])
    increment_exchanges(u["user_id"])
    assert get_user(u["user_id"])["exchanges_used"] == 2


def test_deleting_chat_history_does_not_refund_exchanges():
    """The regression the whole column exists for.

    delete_session removes the session's messages, so a quota derived from
    chat_messages would hand the user a fresh 25 every time they cleared
    their history.
    """
    from app.core.chat_store import add_message, create_session, delete_session
    from app.core.user_store import get_user, increment_exchanges, register_user

    u = register_user(f"q_{uuid.uuid4().hex[:8]}", "correct-horse-battery")
    session = create_session(u["user_id"], "a chat")

    add_message(session["session_id"], "user", "question one")
    increment_exchanges(u["user_id"])

    delete_session(session["session_id"])

    assert get_user(u["user_id"])["exchanges_used"] == 1, (
        "clearing chat history refunded the user's lifetime quota"
    )


# ── enforcement at the endpoints ─────────────────────────────────────

def _sse(payload: dict) -> str:
    import json
    return f"data: {json.dumps(payload)}\n\n"


def _register(client) -> str:
    client.cookies.clear()
    username = f"q_{uuid.uuid4().hex[:10]}"
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": "correct-horse-battery"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["user_id"]


def _spend(user_id: str, n: int) -> None:
    from app.core.user_store import increment_exchanges

    for _ in range(n):
        increment_exchanges(user_id)


def fake_answer_stream(question, chat_history=None, top_k=None, user_id="",
                       session_id="", gated=None):
    async def gen():
        yield _sse({"type": "meta", "sources": [], "images": [],
                    "trace_id": "t", "session_id": session_id})
        yield _sse({"type": "token", "content": "an answer", "attempt": 1})
        yield _sse({"type": "done", "usage": {}, "final_attempt": 1})
    return gen()


def test_the_26th_question_is_refused(client, monkeypatch):
    import app.api.routes.chat as chat_routes

    def must_not_run(**kwargs):
        raise AssertionError("the engine was called despite an exhausted quota")

    monkeypatch.setattr(chat_routes, "ask_stream", must_not_run)
    user_id = _register(client)
    _spend(user_id, 25)

    resp = client.post("/api/v1/chat/stream", json={"question": "one more?"})

    assert resp.status_code == 403
    assert "25" in resp.json()["detail"]
    client.cookies.clear()


def test_an_answered_question_spends_one_exchange(client, monkeypatch):
    import app.api.routes.chat as chat_routes
    from app.core.user_store import get_user

    monkeypatch.setattr(chat_routes, "ask_stream", fake_answer_stream)
    user_id = _register(client)

    assert client.post("/api/v1/chat/stream", json={"question": "hello?"}).status_code == 200

    assert get_user(user_id)["exchanges_used"] == 1
    client.cookies.clear()


def test_a_question_that_never_produced_an_answer_is_not_charged(client, monkeypatch):
    """Checking costs nothing; only a delivered answer spends the budget."""
    import app.api.routes.chat as chat_routes
    from app.core.user_store import get_user

    def explodes(question, chat_history=None, top_k=None, user_id="",
                 session_id="", gated=None):
        async def gen():
            yield _sse({"type": "meta", "sources": [], "images": [],
                        "trace_id": "t", "session_id": session_id})
            raise RuntimeError("azure fell over")
            yield  # pragma: no cover - unreachable, makes this a generator
        return gen()

    monkeypatch.setattr(chat_routes, "ask_stream", explodes)
    user_id = _register(client)

    client.post("/api/v1/chat/stream", json={"question": "hello?"})

    assert get_user(user_id)["exchanges_used"] == 0
    client.cookies.clear()


def test_an_admin_is_not_charged_and_not_refused(client, monkeypatch):
    import app.api.routes.chat as chat_routes
    from app.core.user_store import get_user, set_user_role

    monkeypatch.setattr(chat_routes, "ask_stream", fake_answer_stream)
    user_id = _register(client)
    set_user_role(user_id, "admin")
    _spend(user_id, 25)

    resp = client.post("/api/v1/chat/stream", json={"question": "still fine?"})

    assert resp.status_code == 200
    assert get_user(user_id)["exchanges_used"] == 25, "an admin was charged"
    client.cookies.clear()


def test_the_third_document_is_refused_before_it_is_saved(client, monkeypatch):
    import app.api.routes.documents as doc_routes

    async def must_not_run(*a, **kw):
        raise AssertionError("the upload was saved despite an exhausted quota")

    monkeypatch.setattr(doc_routes, "save_upload", must_not_run)
    monkeypatch.setattr(
        doc_routes, "get_user_documents",
        lambda user_id: [{"doc_id": "a"}, {"doc_id": "b"}],
    )
    _register(client)

    resp = client.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.txt", b"hello there", "text/plain")},
    )

    assert resp.status_code == 403
    assert "Delete one" in resp.json()["detail"]
    client.cookies.clear()


def test_deleting_a_document_frees_a_slot(client, monkeypatch):
    """The held-count semantics: one slot back, not one use back.

    save_upload is replaced by a sentinel rather than letting the upload run:
    a real ingest would index into the session-wide vector store and change
    what every later retrieval test sees. Reaching the sentinel is the proof
    that the quota gate let the request through.
    """
    import app.api.routes.documents as doc_routes

    PAST_THE_GATE = 418

    async def sentinel(*a, **kw):
        raise HTTPException(status_code=PAST_THE_GATE, detail="reached the upload")

    held: list[dict] = [{"doc_id": "a"}, {"doc_id": "b"}]
    monkeypatch.setattr(doc_routes, "get_user_documents", lambda user_id: held)
    monkeypatch.setattr(doc_routes, "save_upload", sentinel)
    _register(client)

    upload = {"file": ("notes.txt", b"hello there", "text/plain")}

    assert client.post("/api/v1/documents/upload", files=upload).status_code == 403

    held.pop()

    assert client.post("/api/v1/documents/upload", files=upload).status_code == PAST_THE_GATE, (
        "a freed slot was not honoured"
    )
    client.cookies.clear()


def test_me_reports_what_is_left(client):
    user_id = _register(client)
    _spend(user_id, 3)

    body = client.get("/api/v1/auth/me").json()

    assert body["exchanges_used"] == 3
    assert body["exchanges_limit"] == 25
    assert body["documents_used"] == 0
    assert body["documents_limit"] == 2
    client.cookies.clear()
