"""Tokens reach the reader before the gate runs, and the gate stops blocking.

A single request measured 3m41s before this: the eval-gated pipeline waited
for generation, scoring, regeneration and a second scoring pass, then sliced
the finished string into 4-character "tokens". The gate itself cannot be made
faster - gpt-5.2 needs ~24s for the faithfulness metric and a smaller model
needs more - so the fix is to stop it blocking.

Azure is stubbed at two boundaries: the streaming client and the two
evaluator functions. Nothing here reaches the network.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import app.core.rag_engine as engine


# ── stubs ────────────────────────────────────────────────────────────

class _FakeDelta:
    def __init__(self, content): self.content = content


class _FakeChoice:
    def __init__(self, content): self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content): self.choices = [_FakeChoice(content)]
    usage = None


class _FakeStream:
    """Async iterator shaped like the Azure streaming response."""

    def __init__(self, pieces): self._pieces = pieces

    def __aiter__(self):
        async def gen():
            for p in self._pieces:
                yield _FakeChunk(p)
        return gen()


class _FakeCompletions:
    def __init__(self, answers):
        self._answers = list(answers)
        self.calls = 0

    async def create(self, **kwargs):
        # Each call streams the next scripted answer, so attempt 2 differs
        # from attempt 1 the way a real regeneration would.
        idx = min(self.calls, len(self._answers) - 1)
        self.calls += 1
        return _FakeStream(self._answers[idx])


class _FakeClient:
    def __init__(self, answers):
        self.chat = type("chat", (), {"completions": _FakeCompletions(answers)})()


def _install(
    monkeypatch,
    answers=(["draft "], ["final answer"]),
    faithfulness=0.9,
    context_precision=0.8,
):
    """Point the engine at fakes; return the client so calls can be counted."""
    client = _FakeClient(answers)
    monkeypatch.setattr(engine, "_get_async_client", lambda: client)
    monkeypatch.setattr(
        engine, "_build_context",
        lambda q, top_k=None, user_id="": ("ctx text", [{"source": "d.pdf", "page": 1,
                                                        "chunk_index": 0,
                                                        "content_type": "text"}], []),
    )

    def _faith(question, answer, contexts):
        if isinstance(faithfulness, Exception):
            raise faithfulness
        return faithfulness

    monkeypatch.setattr(engine, "evaluate_faithfulness_sync", _faith)
    monkeypatch.setattr(
        engine, "evaluate_context_precision_sync",
        lambda question, contexts: context_precision,
    )
    monkeypatch.setattr(engine, "evaluate_query_async", lambda **kw: None)
    return client


def _collect(gated=True, **kw):
    """Drive the async generator to completion and return parsed events."""
    async def run():
        out = []
        async for chunk in engine.ask_stream(
            question="what changed?", user_id="u1", session_id="s1", gated=gated, **kw
        ):
            assert chunk.startswith("data: "), chunk
            out.append(json.loads(chunk[6:]))
        return out

    return asyncio.run(run())


def _types(events): return [e["type"] for e in events]


def _text(events, attempt):
    return "".join(
        e["content"] for e in events
        if e["type"] == "token" and e.get("attempt") == attempt
    )


# ── the core claim ───────────────────────────────────────────────────

def test_a_token_arrives_before_any_eval_event(monkeypatch):
    """The whole point: the reader is not held behind the gate."""
    _install(monkeypatch)
    types = _types(_collect())

    assert "token" in types and "eval" in types
    assert types.index("token") < types.index("eval")


def test_meta_precedes_the_first_token(monkeypatch):
    """Sources must be renderable while tokens stream."""
    _install(monkeypatch)
    types = _types(_collect())
    assert types.index("meta") < types.index("token")


# ── gate outcomes ────────────────────────────────────────────────────

def test_a_failing_gate_replaces_the_draft(monkeypatch):
    _install(monkeypatch, answers=(["bad draft"], ["grounded answer"]),
             faithfulness=0.2, context_precision=0.8)
    events = _collect()

    assert "replace" in _types(events)
    assert _text(events, 1) == "bad draft"
    assert _text(events, 2) == "grounded answer"
    assert events[-1]["type"] == "done"
    assert events[-1]["final_attempt"] == 2


def test_a_passing_gate_leaves_one_attempt(monkeypatch):
    client = _install(monkeypatch, faithfulness=0.9, context_precision=0.8)
    events = _collect()

    assert "replace" not in _types(events)
    assert events[-1]["final_attempt"] == 1
    assert client.chat.completions.calls == 1, "regenerated despite passing"


def test_zero_context_precision_does_not_regenerate(monkeypatch):
    """Measured: this path cost ~94s and could not have helped.

    A stricter prompt against identical, irrelevant context cannot raise
    grounding. Report the retrieval failure instead of retrying.
    """
    client = _install(monkeypatch, faithfulness=0.2, context_precision=0.0)
    events = _collect()

    verdicts = [e["verdict"] for e in events if e["type"] == "eval"]
    assert verdicts == ["retrieval_failed"]
    assert "replace" not in _types(events)
    assert client.chat.completions.calls == 1


def test_unknown_context_precision_is_not_treated_as_retrieval_failure(monkeypatch):
    """None means the metric errored; 0.0 means retrieval genuinely missed."""
    _install(monkeypatch, faithfulness=0.2, context_precision=None)
    events = _collect()

    verdicts = [e["verdict"] for e in events if e["type"] == "eval"]
    assert verdicts[0] == "rejected"
    assert "replace" in _types(events)


def test_a_raising_gate_keeps_the_draft(monkeypatch):
    """An evaluation failure must never cost the reader their answer."""
    _install(monkeypatch, answers=(["the draft"],),
             faithfulness=RuntimeError("ragas exploded"))
    events = _collect()

    assert [e["verdict"] for e in events if e["type"] == "eval"] == ["unscored"]
    assert _text(events, 1) == "the draft"
    assert events[-1]["final_attempt"] == 1


def test_the_regenerated_answer_is_not_scored_before_done(monkeypatch):
    """The uninstrumented ~74s second faithfulness call is gone."""
    calls = []

    client = _install(monkeypatch, answers=(["bad"], ["better"]),
                      faithfulness=0.2, context_precision=0.8)

    original = engine.evaluate_faithfulness_sync

    def counting(question, answer, contexts):
        calls.append(answer)
        return original(question, answer, contexts)

    monkeypatch.setattr(engine, "evaluate_faithfulness_sync", counting)
    _collect()

    assert calls == ["bad"], f"gate ran on the replacement too: {calls}"


# ── gating disabled ──────────────────────────────────────────────────

def test_gating_off_streams_without_evaluating(monkeypatch):
    _install(monkeypatch)
    events = _collect(gated=False)

    types = _types(events)
    assert "eval" not in types
    assert "replace" not in types
    assert types[0] == "meta"
    assert types[-1] == "done"
    assert _text(events, 1) == "draft "


def test_stage_events_describe_the_phase(monkeypatch):
    _install(monkeypatch, answers=(["bad"], ["better"]),
             faithfulness=0.2, context_precision=0.8)
    stages = [e["stage"] for e in _collect() if e["type"] == "stage"]

    assert stages == ["generating", "scoring", "regenerating"]
