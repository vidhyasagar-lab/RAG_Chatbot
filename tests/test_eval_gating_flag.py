"""EVAL_GATING_ENABLED chooses which pipeline /chat/stream runs.

Both pipelines call Azure OpenAI, so each is replaced at that boundary by a
stub emitting the same SSE event shapes the real one does (captured from a
live run: meta -> [eval ->] token -> done).
"""

from __future__ import annotations

import json
import uuid

import pytest


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _meta(session_id: str) -> str:
    return _sse({"type": "meta", "sources": [], "images": [], "trace_id": "t-1", "session_id": session_id})


def fake_plain_stream(question, chat_history=None, top_k=None, user_id="", session_id=""):
    """Sync generator, like the real ask_stream."""
    yield _meta(session_id)
    yield _sse({"type": "token", "content": "plain "})
    yield _sse({"type": "token", "content": "answer"})
    yield _sse({"type": "done", "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}})


async def fake_gated_stream(question, chat_history=None, top_k=None, user_id="", session_id=""):
    """Async generator, like the real ask_with_eval."""
    yield _meta(session_id)
    yield _sse({"type": "eval", "scores": {"context_precision": 0.9, "faithfulness": 0.95, "threshold": 0.75, "passed": True}})
    yield _sse({"type": "token", "content": "gated answer"})
    yield _sse({"type": "done", "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}})


@pytest.fixture
def signed_in(client, monkeypatch):
    import app.api.routes.chat as chat_routes

    monkeypatch.setattr(chat_routes, "ask_stream", fake_plain_stream)
    monkeypatch.setattr(chat_routes, "ask_with_eval", fake_gated_stream)
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


def test_gating_off_streams_the_plain_pipeline(signed_in, monkeypatch):
    _set_gating(monkeypatch, False)
    events = _events(signed_in)
    assert [e["type"] for e in events] == ["meta", "token", "token", "done"]
    assert "".join(e["content"] for e in events if e["type"] == "token") == "plain answer"


def test_gating_on_streams_the_eval_gated_pipeline(signed_in, monkeypatch):
    _set_gating(monkeypatch, True)
    events = _events(signed_in)
    assert [e["type"] for e in events] == ["meta", "eval", "token", "done"]


def test_gating_off_still_saves_the_answer_to_history(signed_in, monkeypatch):
    _set_gating(monkeypatch, False)
    session_id = _events(signed_in)[0]["session_id"]

    resp = signed_in.get(f"/api/v1/chat/sessions/{session_id}")
    assert resp.status_code == 200
    messages = resp.json()["messages"]
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "what changed?"),
        ("assistant", "plain answer"),
    ]
