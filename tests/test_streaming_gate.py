"""Tokens reach the reader before the gate runs, and the gate stops blocking.

A single request measured 3m41s before this: the eval-gated pipeline waited
for generation, scoring, regeneration and a second scoring pass, then sliced
the finished string into 4-character "tokens". The gate itself cannot be made
faster - gpt-5.2 needs ~24s for the faithfulness metric and a smaller model
needs more - so the fix is to stop it blocking.

It used to run inside the stream, after the answer. Everything below the
answer - sources, figures, the next question - waited for `done`, which the
gate held for ~36s. It now runs as a background task after `done`, and its
verdict (and any revised answer) is fetched from /chat/gate/{trace_id}.

Azure is stubbed at two boundaries: the streaming client and the two
evaluator functions. Nothing here reaches the network.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.documents import Document

import app.core.rag_engine as engine


# ── stubs ────────────────────────────────────────────────────────────

class _FakeDelta:
    def __init__(self, content): self.content = content


class _FakeChoice:
    def __init__(self, content): self.delta = _FakeDelta(content)


class _FakeUsage:
    prompt_tokens = 100
    completion_tokens = 20
    total_tokens = 120


class _FakeChunk:
    def __init__(self, content, usage=None):
        self.choices = [_FakeChoice(content)]
        self.usage = usage


class _FakeStream:
    """Async iterator shaped like the Azure streaming response.

    Records whether it was closed, because an unclosed stream leaves the HTTP
    response to Azure open and the socket in CLOSE_WAIT.
    """

    def __init__(self, pieces):
        self._pieces = pieces
        self.closed = False

    def __aiter__(self):
        async def gen():
            for i, p in enumerate(self._pieces):
                last = i == len(self._pieces) - 1
                yield _FakeChunk(p, usage=_FakeUsage() if last else None)
        return gen()

    async def close(self):
        self.closed = True


class _FakeCompletions:
    def __init__(self, answers):
        self._answers = list(answers)
        self.calls = 0
        self.kwargs: list[dict] = []
        self.streams: list[_FakeStream] = []

    async def create(self, **kwargs):
        # Each call streams the next scripted answer, so attempt 2 differs
        # from attempt 1 the way a real regeneration would.
        idx = min(self.calls, len(self._answers) - 1)
        self.calls += 1
        self.kwargs.append(kwargs)
        stream = _FakeStream(self._answers[idx])
        self.streams.append(stream)
        return stream


class _FakeClient:
    def __init__(self, answers):
        self.chat = type("chat", (), {"completions": _FakeCompletions(answers)})()


def _install(
    monkeypatch,
    answers=(["draft "], ["final answer"]),
    faithfulness=0.9,
    context_precision=0.8,
    followups=(),
):
    """Point the engine at fakes; return the client so calls can be counted."""
    client = _FakeClient(answers)
    monkeypatch.setattr(engine, "_get_async_client", lambda: client)
    monkeypatch.setattr(
        engine, "hybrid_search",
        lambda q, k=None, user_id="": [Document(
            page_content="ctx text",
            metadata={"source": "d.pdf", "page": 1, "content_type": "text"})],
    )
    monkeypatch.setattr(engine, "plan_followups", lambda q, docs: list(followups))

    def _faith(question, answer, contexts):
        if isinstance(faithfulness, Exception):
            raise faithfulness
        return faithfulness

    monkeypatch.setattr(engine, "evaluate_faithfulness_sync", _faith)
    monkeypatch.setattr(
        engine, "evaluate_context_precision_sync",
        lambda question, contexts, answer: context_precision,
    )
    monkeypatch.setattr(engine, "evaluate_query_async", lambda **kw: None)
    monkeypatch.setattr(engine, "create_trace", lambda **kw: _Recorded([], "trace"))
    _saved(monkeypatch)
    return client


def _saved(monkeypatch) -> dict:
    """Capture what the background gate persists, instead of writing SQLite."""
    saved = {"results": [], "revisions": []}
    monkeypatch.setattr(engine, "save_gate_result",
                        lambda trace_id, user_id, result: saved["results"].append(result))
    monkeypatch.setattr(
        engine, "revise_answer_message",
        lambda trace_id, revised, eval_meta: saved["revisions"].append(
            (trace_id, revised, eval_meta["verdict"])),
    )
    return saved


def _collect(gated=True, **kw):
    """Drive the stream to `done`, then let the background gate finish.

    asyncio.run cancels whatever is still pending when its coroutine returns,
    so the gate tasks are awaited inside it.
    """
    async def run():
        out = []
        async for chunk in engine.ask_stream(
            question="what changed?", user_id="u1", session_id="s1", gated=gated, **kw
        ):
            assert chunk.startswith("data: "), chunk
            out.append(json.loads(chunk[6:]))
        await engine.wait_for_gates()
        return out

    return asyncio.run(run())


def _types(events): return [e["type"] for e in events]


def _text(events, attempt):
    return "".join(
        e["content"] for e in events
        if e["type"] == "token" and e.get("attempt") == attempt
    )


def _gate(saved) -> dict:
    assert len(saved["results"]) == 1, saved["results"]
    return saved["results"][0]


# ── the core claim ───────────────────────────────────────────────────

def test_the_stream_ends_with_the_answer_not_the_gate(monkeypatch):
    """Sources, figures and the next question all wait for `done`. With the
    gate inside the stream that was ~36s after the answer; now it is none."""
    _install(monkeypatch)
    events = _collect()

    types = _types(events)
    assert "eval" not in types and "replace" not in types
    assert types[-1] == "done"
    assert events[-1]["gate"] == "pending"
    assert events[-1]["final_attempt"] == 1


def test_done_does_not_wait_for_a_slow_gate(monkeypatch):
    import time

    _install(monkeypatch)
    monkeypatch.setattr(engine, "evaluate_faithfulness_sync",
                        lambda question, answer, contexts: time.sleep(1.0) or 0.9)

    async def run():
        start = time.perf_counter()
        async for chunk in engine.ask_stream(question="q", user_id="u", session_id="s", gated=True):
            if json.loads(chunk[6:])["type"] == "done":
                elapsed = time.perf_counter() - start
        await engine.wait_for_gates()
        return elapsed

    assert asyncio.run(run()) < 0.5


def test_meta_precedes_the_first_token(monkeypatch):
    """Sources must be renderable while tokens stream."""
    _install(monkeypatch)
    types = _types(_collect())
    assert types.index("meta") < types.index("token")


# ── gate outcomes, now reported after the stream ─────────────────────

def test_a_failing_gate_produces_a_revised_answer(monkeypatch):
    _install(monkeypatch, answers=(["bad draft"], ["grounded answer"]),
             faithfulness=0.2, context_precision=0.8)
    saved = _saved(monkeypatch)
    events = _collect()

    assert _text(events, 1) == "bad draft"
    assert _gate(saved)["verdict"] == "rejected"
    assert _gate(saved)["revised_answer"] == "grounded answer"
    assert saved["revisions"] == [("trace-x", "grounded answer", "rejected")]


def test_a_passing_gate_leaves_one_attempt(monkeypatch):
    client = _install(monkeypatch, faithfulness=0.9, context_precision=0.8)
    saved = _saved(monkeypatch)
    _collect()

    assert _gate(saved)["verdict"] == "passed"
    assert _gate(saved)["revised_answer"] is None
    assert client.chat.completions.calls == 1, "regenerated despite passing"


def test_zero_context_precision_does_not_regenerate(monkeypatch):
    """Measured: this path cost ~94s and could not have helped."""
    client = _install(monkeypatch, faithfulness=0.2, context_precision=0.0)
    saved = _saved(monkeypatch)
    _collect()

    assert _gate(saved)["verdict"] == "retrieval_failed"
    assert client.chat.completions.calls == 1


def test_unknown_context_precision_is_not_treated_as_retrieval_failure(monkeypatch):
    """None means the metric errored; 0.0 means retrieval genuinely missed."""
    _install(monkeypatch, faithfulness=0.2, context_precision=None)
    saved = _saved(monkeypatch)
    _collect()

    assert _gate(saved)["verdict"] == "rejected"


def test_a_raising_gate_keeps_the_draft(monkeypatch):
    """An evaluation failure must never cost the reader their answer."""
    _install(monkeypatch, answers=(["the draft"],),
             faithfulness=RuntimeError("ragas exploded"))
    saved = _saved(monkeypatch)
    events = _collect()

    assert _gate(saved)["verdict"] == "unscored"
    assert _gate(saved)["revised_answer"] is None
    assert _text(events, 1) == "the draft"


def test_a_crashing_gate_still_reports_a_result(monkeypatch):
    """The badge polls for a result; without one it spins until it gives up."""
    _install(monkeypatch)
    saved = _saved(monkeypatch)

    def boom(*a, **kw):
        raise RuntimeError("verdict logic broke")

    monkeypatch.setattr(engine, "_decide_verdict", boom)
    _collect()

    assert _gate(saved)["verdict"] == "unscored"


def test_a_slow_precision_metric_does_not_discard_a_finished_faithfulness(monkeypatch):
    """Precision makes one judge call per context, so it is the metric that
    overruns. A finished faithfulness score must survive its timeout."""
    import time

    from app.config import get_settings

    _install(monkeypatch, faithfulness=0.9)
    saved = _saved(monkeypatch)
    monkeypatch.setattr(engine, "evaluate_context_precision_sync",
                        lambda question, contexts, answer: time.sleep(1.5) or 0.8)
    monkeypatch.setattr(get_settings(), "eval_timeout_seconds", 0.3)
    _collect()

    assert _gate(saved)["verdict"] == "passed"
    assert _gate(saved)["faithfulness"] == 0.9
    assert _gate(saved)["context_precision"] is None


def test_the_regenerated_answer_is_not_gated_again(monkeypatch):
    calls = []
    _install(monkeypatch, answers=(["bad"], ["better"]),
             faithfulness=0.2, context_precision=0.8)
    original = engine.evaluate_faithfulness_sync

    def counting(question, answer, contexts):
        calls.append(answer)
        return original(question, answer, contexts)

    monkeypatch.setattr(engine, "evaluate_faithfulness_sync", counting)
    _collect()

    assert calls == ["bad"], f"gate ran on the replacement too: {calls}"


def test_the_final_answer_is_scored_in_the_background(monkeypatch):
    _install(monkeypatch, answers=(["bad"], ["better"]),
             faithfulness=0.2, context_precision=0.8)
    scored = []
    monkeypatch.setattr(engine, "evaluate_query_async", lambda **kw: scored.append(kw["answer"]))
    _collect()

    assert scored == ["better"]


def test_an_empty_regeneration_does_not_destroy_the_answer(monkeypatch):
    """A content filter or an empty completion must not replace a usable draft."""
    _install(monkeypatch, answers=(["a usable draft"], [""]),
             faithfulness=0.2, context_precision=0.8)
    saved = _saved(monkeypatch)
    _collect()

    assert _gate(saved)["revised_answer"] is None
    assert saved["revisions"] == [("trace-x", None, "rejected")]


def test_context_precision_is_scored_against_the_real_answer(monkeypatch):
    """"Without reference" means without a ground truth, not without the answer."""
    seen = {}
    _install(monkeypatch, answers=(["the streamed answer"],))
    monkeypatch.setattr(
        engine, "evaluate_context_precision_sync",
        lambda question, contexts, answer: seen.update(answer=answer) or 0.8,
    )
    _collect()

    assert seen.get("answer") == "the streamed answer"


# ── gating disabled ──────────────────────────────────────────────────

def test_gating_off_streams_without_evaluating(monkeypatch):
    _install(monkeypatch)
    saved = _saved(monkeypatch)
    events = _collect(gated=False)

    assert _types(events)[0] == "meta"
    assert _types(events)[-1] == "done"
    assert "gate" not in events[-1]
    assert saved["results"] == []
    assert _text(events, 1) == "draft "


def test_gating_off_still_scores_the_answer_in_the_background(monkeypatch):
    """The dashboard's scores come from here. Without it they stop silently."""
    _install(monkeypatch)
    scored = []
    monkeypatch.setattr(engine, "evaluate_query_async", lambda **kw: scored.append(kw))
    _collect(gated=False)

    assert len(scored) == 1
    assert scored[0]["answer"] == "draft "


# ── plumbing ─────────────────────────────────────────────────────────

def test_usage_is_requested_and_reported(monkeypatch):
    client = _install(monkeypatch)
    events = _collect()

    assert client.chat.completions.kwargs[0].get("stream_options") == {"include_usage": True}
    assert events[-1]["usage"].get("total_tokens") == 120


def test_every_answer_stream_is_closed(monkeypatch):
    """An unclosed stream leaves the HTTP response to Azure open - including
    the regeneration, which is now read in the background."""
    client = _install(monkeypatch, answers=(["bad"], ["better"]),
                      faithfulness=0.2, context_precision=0.8)
    _collect()

    assert len(client.chat.completions.streams) == 2
    assert all(s.closed for s in client.chat.completions.streams)


def test_a_long_answer_is_cut_on_a_boundary_not_mid_word(monkeypatch):
    seen = {}
    long_answer = ("The ingest pipeline reads each document. " * 400)  # ~16k chars

    _install(monkeypatch, answers=([long_answer],))
    monkeypatch.setattr(
        engine, "evaluate_faithfulness_sync",
        lambda question, answer, contexts: seen.update(scored=answer) or 0.9,
    )
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "eval_max_answer_chars", 6000)
    _collect()

    scored = seen["scored"]
    assert len(scored) <= 6000
    assert scored.endswith(" ") or scored.endswith(".") or scored == long_answer


def test_the_async_client_is_reused_across_requests():
    engine._reset_async_client()
    try:
        assert engine._get_async_client() is engine._get_async_client()
    finally:
        engine._reset_async_client()


def test_the_gate_does_not_run_on_the_default_thread_pool():
    assert engine._EVAL_EXECUTOR is not None
    assert engine._EVAL_EXECUTOR._max_workers >= 2


def test_concurrent_gated_requests_both_complete(monkeypatch):
    """The concurrency bound must not deadlock two simultaneous readers."""
    _install(monkeypatch)
    saved = _saved(monkeypatch)

    async def run_two():
        async def one():
            return [
                json.loads(c[6:])
                async for c in engine.ask_stream(question="q", user_id="u",
                                                 session_id="s", gated=True)
            ]
        results = await asyncio.gather(one(), one())
        await engine.wait_for_gates()
        return results

    first, second = asyncio.run(run_two())
    assert first[-1]["type"] == "done"
    assert second[-1]["type"] == "done"
    assert len(saved["results"]) == 2


# ── spans and the trace are always closed ────────────────────────────

class _Recorded:
    """A trace or span that remembers whether it was ended."""

    def __init__(self, log: list, name: str):
        self.name, self.ended, self.id, self._log = name, False, "trace-x", log
        log.append(self)

    def span(self, **kw): return _Recorded(self._log, kw.get("name", "?"))
    generation = span
    def update(self, **kw): pass
    def end(self, **kw): self.ended = True


def _recording(monkeypatch) -> list[_Recorded]:
    opened: list[_Recorded] = []
    monkeypatch.setattr(engine, "create_trace", lambda **kw: _Recorded(opened, "trace"))
    return opened


def test_leaving_while_the_draft_streams_closes_everything(monkeypatch):
    """An unended span shows in Langfuse as a request still running, forever."""
    _install(monkeypatch, answers=(["bad"], ["better"]),
             faithfulness=0.2, context_precision=0.8)
    opened = _recording(monkeypatch)

    async def run():
        agen = engine.ask_stream(question="q", user_id="u", session_id="s", gated=True)
        async for chunk in agen:
            if json.loads(chunk[6:])["type"] == "token":
                break
        await agen.aclose()
        await engine.wait_for_gates()

    asyncio.run(run())
    assert [o.name for o in opened if not o.ended] == []


def test_leaving_mid_draft_does_not_start_a_gate(monkeypatch):
    """Nobody is there to read the result, and the draft was never finished."""
    client = _install(monkeypatch, answers=(["bad"], ["better"]),
                      faithfulness=0.2, context_precision=0.8)
    saved = _saved(monkeypatch)
    _recording(monkeypatch)

    async def run():
        agen = engine.ask_stream(question="q", user_id="u", session_id="s", gated=True)
        async for chunk in agen:
            if json.loads(chunk[6:])["type"] == "token":
                break
        await agen.aclose()
        await engine.wait_for_gates()

    asyncio.run(run())
    assert client.chat.completions.calls == 1
    assert saved["results"] == []


@pytest.mark.parametrize("outcome", [
    dict(faithfulness=0.9, context_precision=0.8),          # passed
    dict(faithfulness=0.2, context_precision=0.8),          # rejected, regenerated
    dict(faithfulness=RuntimeError("x"), context_precision=0.8),  # unscored
], ids=["passed", "rejected", "unscored"])
def test_the_background_gate_closes_every_span_and_the_trace(monkeypatch, outcome):
    _install(monkeypatch, answers=(["bad"], ["better"]), **outcome)
    opened = _recording(monkeypatch)
    _collect()

    assert [o.name for o in opened if not o.ended] == []


# ── follow-up search ─────────────────────────────────────────────────

def test_a_second_search_is_announced_before_the_sources(monkeypatch):
    """The wait is explained, and the sources shown include round two's."""
    _install(monkeypatch, followups=["ranking table"])
    events = _collect()
    types = _types(events)

    searching = [i for i, e in enumerate(events) if e.get("stage") == "searching"]
    assert len(searching) == 1
    assert searching[0] < types.index("meta")


def test_no_second_search_means_no_searching_stage(monkeypatch):
    _install(monkeypatch)
    assert all(e.get("stage") != "searching" for e in _collect())


def test_the_stream_only_announces_generation(monkeypatch):
    """Scoring and regenerating happen after the stream now."""
    _install(monkeypatch, answers=(["bad"], ["better"]),
             faithfulness=0.2, context_precision=0.8)
    stages = [e["stage"] for e in _collect() if e["type"] == "stage"]

    assert stages == ["generating"]
