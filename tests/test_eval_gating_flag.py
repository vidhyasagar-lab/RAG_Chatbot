"""EVAL_GATING_ENABLED changes what /chat/stream emits, not whether it streams.

Before 2026-09-23 the flag chose between two pipelines and only one of them
streamed, so turning gating on silently turned streaming off. Now there is one
pipeline: the flag decides whether the gate runs after the tokens, and tokens
go out either way.

ask_stream calls Azure, so it is replaced here by a stub emitting the same SSE
shapes the real one does.
"""

from __future__ import annotations

import json
import uuid

import pytest


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _meta(session_id: str) -> str:
    return _sse({"type": "meta", "sources": [], "images": [],
                 "trace_id": "t-1", "session_id": session_id})


def fake_stream(question, chat_history=None, top_k=None, user_id="",
                session_id="", gated=None):
    """Async generator matching the real ask_stream's contract."""
    from app.config import get_settings

    if gated is None:
        gated = get_settings().eval_gating_enabled

    async def gen():
        yield _meta(session_id)
        yield _sse({"type": "stage", "stage": "generating", "attempt": 1})
        yield _sse({"type": "token", "content": "plain ", "attempt": 1})
        yield _sse({"type": "token", "content": "answer", "attempt": 1})
        if gated:
            yield _sse({"type": "stage", "stage": "scoring", "attempt": 1})
            yield _sse({
                "type": "eval", "attempt": 1, "verdict": "passed",
                "scores": {"context_precision": 0.9, "faithfulness": 0.95,
                           "threshold": 0.5, "passed": True},
            })
        yield _sse({"type": "done", "usage": {"total_tokens": 3}, "final_attempt": 1})

    return gen()


def fake_rejecting_stream(question, chat_history=None, top_k=None, user_id="",
                          session_id="", gated=None):
    """A gate that rejects the draft and streams a replacement."""
    async def gen():
        yield _meta(session_id)
        yield _sse({"type": "stage", "stage": "generating", "attempt": 1})
        yield _sse({"type": "token", "content": "ungrounded draft", "attempt": 1})
        yield _sse({"type": "stage", "stage": "scoring", "attempt": 1})
        yield _sse({
            "type": "eval", "attempt": 1, "verdict": "rejected",
            "scores": {"context_precision": 0.8, "faithfulness": 0.2,
                       "threshold": 0.5, "passed": False},
        })
        yield _sse({"type": "replace", "reason": "faithfulness 0.20 < 0.50"})
        yield _sse({"type": "stage", "stage": "regenerating", "attempt": 2})
        yield _sse({"type": "token", "content": "grounded answer", "attempt": 2})
        yield _sse({"type": "done", "usage": {"total_tokens": 6}, "final_attempt": 2})

    return gen()


def fake_interrupted_stream(question, chat_history=None, top_k=None, user_id="",
                            session_id="", gated=None):
    """Tokens complete, then the stream ends without `done`.

    This is what a reader sees when they hit Stop, navigate away or close the
    tab during the quality gate - a window that is now 25-120s wide, because
    `done` is emitted after the gate rather than after the last token.
    """
    async def gen():
        yield _meta(session_id)
        yield _sse({"type": "stage", "stage": "generating", "attempt": 1})
        yield _sse({"type": "token", "content": "a complete answer", "attempt": 1})
        yield _sse({"type": "stage", "stage": "scoring", "attempt": 1})
        # ...and then nothing. No done, no error.

    return gen()


@pytest.fixture
def signed_in(client, monkeypatch):
    import app.api.routes.chat as chat_routes

    monkeypatch.setattr(chat_routes, "ask_stream", fake_stream)
    client.cookies.clear()
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": f"gate_{uuid.uuid4().hex[:10]}", "password": "correct-horse-battery"},
    )
    assert resp.status_code == 201
    yield client
    client.cookies.clear()


def _events(client) -> list[dict]:
    resp = client.post("/api/v1/chat/stream", json={"question": "what changed?"})
    assert resp.status_code == 200
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def _set_gating(monkeypatch, enabled: bool) -> None:
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "eval_gating_enabled", enabled)


def test_gating_off_streams_without_an_eval_event(signed_in, monkeypatch):
    _set_gating(monkeypatch, False)
    events = _events(signed_in)
    assert [e["type"] for e in events] == ["meta", "stage", "token", "token", "done"]


def test_gating_on_still_streams_tokens_before_the_eval(signed_in, monkeypatch):
    """The regression that motivated this work: gating used to stop streaming."""
    _set_gating(monkeypatch, True)
    types = [e["type"] for e in _events(signed_in)]
    assert types.index("token") < types.index("eval")


def test_gating_off_still_saves_the_answer_to_history(signed_in, monkeypatch):
    _set_gating(monkeypatch, False)
    session_id = _events(signed_in)[0]["session_id"]

    resp = signed_in.get(f"/api/v1/chat/sessions/{session_id}")
    assert resp.status_code == 200
    assert [(m["role"], m["content"]) for m in resp.json()["messages"]] == [
        ("user", "what changed?"),
        ("assistant", "plain answer"),
    ]


def test_an_answer_survives_a_stream_that_ends_before_done(signed_in, monkeypatch):
    """The reader saw a finished answer; it must not vanish from their history.

    `done` now arrives after the gate, so anything that ends the connection
    during scoring - Stop, navigation, a closed tab, a sleeping laptop - used
    to drop a complete answer the route was already holding in memory.
    """
    import app.api.routes.chat as chat_routes

    monkeypatch.setattr(chat_routes, "ask_stream", fake_interrupted_stream)
    _set_gating(monkeypatch, True)
    session_id = _events(signed_in)[0]["session_id"]

    resp = signed_in.get(f"/api/v1/chat/sessions/{session_id}")
    assert resp.status_code == 200
    assert [(m["role"], m["content"]) for m in resp.json()["messages"]] == [
        ("user", "what changed?"),
        ("assistant", "a complete answer"),
    ]


def test_an_interrupted_answer_is_saved_only_once(signed_in, monkeypatch):
    """The `done` path and the fallback must not both write."""
    import app.api.routes.chat as chat_routes

    monkeypatch.setattr(chat_routes, "ask_stream", fake_stream)
    _set_gating(monkeypatch, True)
    session_id = _events(signed_in)[0]["session_id"]

    resp = signed_in.get(f"/api/v1/chat/sessions/{session_id}")
    assistant = [m for m in resp.json()["messages"] if m["role"] == "assistant"]
    assert len(assistant) == 1, f"saved {len(assistant)} times"


def test_only_the_surviving_answer_reaches_history(signed_in, monkeypatch):
    """A rejected draft was shown as a demonstration, not offered as an answer."""
    import app.api.routes.chat as chat_routes

    monkeypatch.setattr(chat_routes, "ask_stream", fake_rejecting_stream)
    _set_gating(monkeypatch, True)
    session_id = _events(signed_in)[0]["session_id"]

    resp = signed_in.get(f"/api/v1/chat/sessions/{session_id}")
    assert resp.status_code == 200
    stored = [(m["role"], m["content"]) for m in resp.json()["messages"]]
    assert stored == [("user", "what changed?"), ("assistant", "grounded answer")]
    assert "ungrounded draft" not in stored[1][1]
