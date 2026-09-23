"""A reopened chat shows what the answer was built on, not just its text.

/chat/stream used to store only the answer text, so a chat reopened from
history lost its sources, figures, trace id and gate verdict. They are now
stored with the assistant message and returned by GET /chat/sessions/{id}.
"""

from __future__ import annotations

import json
import uuid

import pytest

SOURCES = [{"source": "uploads/u1/report.pdf", "page": 4, "chunk_index": 1, "content_type": "text"}]
IMAGES = [{"path": "uploads/extracted/report_p4_chart.png", "page": 4, "source": "report.pdf", "content_type": "chart"}]


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _meta(session_id: str, trace_id: str = "trace-9") -> str:
    return _sse({"type": "meta", "sources": SOURCES, "images": IMAGES, "trace_id": trace_id, "session_id": session_id})


def gated_stream(question, chat_history=None, top_k=None, user_id="", session_id="", gated=None):
    async def gen():
        yield _meta(session_id)
        yield _sse({"type": "token", "content": "Revenue grew [1].", "attempt": 1})
        yield _sse({
            "type": "eval", "attempt": 1, "verdict": "passed",
            "scores": {"context_precision": 0.8, "faithfulness": 0.9, "threshold": 0.5, "passed": True},
        })
        yield _sse({"type": "done", "final_attempt": 1})

    return gen()


def replaced_stream(question, chat_history=None, top_k=None, user_id="", session_id="", gated=None):
    async def gen():
        yield _meta(session_id)
        yield _sse({"type": "token", "content": "draft", "attempt": 1})
        yield _sse({
            "type": "eval", "attempt": 1, "verdict": "rejected",
            "scores": {"context_precision": 0.8, "faithfulness": 0.2, "threshold": 0.5, "passed": False},
        })
        yield _sse({"type": "replace", "reason": "faithfulness 0.20 < 0.50"})
        yield _sse({"type": "token", "content": "grounded", "attempt": 2})
        yield _sse({
            "type": "eval", "attempt": 2, "verdict": "passed",
            "scores": {"context_precision": 0.8, "faithfulness": 0.95, "threshold": 0.5, "passed": True},
        })
        yield _sse({"type": "done", "final_attempt": 2})

    return gen()


def plain_stream(question, chat_history=None, top_k=None, user_id="", session_id="", gated=None):
    async def gen():
        yield _meta(session_id)
        yield _sse({"type": "token", "content": "Hi", "attempt": 1})
        yield _sse({"type": "done", "final_attempt": 1})

    return gen()


@pytest.fixture
def signed_in(client):
    client.cookies.clear()
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": f"meta_{uuid.uuid4().hex[:10]}", "password": "correct-horse-battery"},
    )
    assert resp.status_code == 201
    yield client
    client.cookies.clear()


def _ask_and_reopen(client, monkeypatch, stream) -> list[dict]:
    import app.api.routes.chat as chat_routes

    monkeypatch.setattr(chat_routes, "ask_stream", stream)
    resp = client.post("/api/v1/chat/stream", json={"question": "What changed?"})
    assert resp.status_code == 200
    first = json.loads(next(line[6:] for line in resp.text.splitlines() if line.startswith("data: ")))
    detail = client.get(f"/api/v1/chat/sessions/{first['session_id']}")
    assert detail.status_code == 200
    return detail.json()["messages"]


def test_a_reopened_answer_keeps_its_sources_figures_and_trace(signed_in, monkeypatch):
    user, answer = _ask_and_reopen(signed_in, monkeypatch, gated_stream)

    assert user == {"role": "user", "content": "What changed?"}
    assert answer["content"] == "Revenue grew [1]."
    assert answer["sources"] == SOURCES
    assert answer["images"] == IMAGES
    assert answer["trace_id"] == "trace-9"


def test_a_reopened_answer_keeps_the_gate_verdict(signed_in, monkeypatch):
    _, answer = _ask_and_reopen(signed_in, monkeypatch, gated_stream)
    assert answer["eval"] == {
        "context_precision": 0.8, "faithfulness": 0.9, "threshold": 0.5, "passed": True,
        "verdict": "passed", "attempt": 1,
    }


def test_the_stored_verdict_is_the_one_for_the_answer_that_stands(signed_in, monkeypatch):
    """A rejected draft's scores must not be shown against its replacement."""
    _, answer = _ask_and_reopen(signed_in, monkeypatch, replaced_stream)
    assert answer["content"] == "grounded"
    assert answer["eval"]["faithfulness"] == 0.95
    assert answer["eval"]["attempt"] == 2


def test_an_ungated_answer_has_no_verdict(signed_in, monkeypatch):
    _, answer = _ask_and_reopen(signed_in, monkeypatch, plain_stream)
    assert answer["sources"] == SOURCES
    assert answer.get("eval") is None


def test_messages_stored_before_this_change_still_load(signed_in):
    """Rows written before the meta column existed have NULL there."""
    from app.core.chat_store import add_message, create_session

    me = signed_in.get("/api/v1/auth/me").json()
    session = create_session(me["user_id"], "old chat")
    add_message(session["session_id"], "user", "old question")
    add_message(session["session_id"], "assistant", "old answer")

    resp = signed_in.get(f"/api/v1/chat/sessions/{session['session_id']}")
    assert resp.status_code == 200
    assert resp.json()["messages"] == [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
    ]
