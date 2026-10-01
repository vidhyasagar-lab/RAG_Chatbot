"""Spend, tokens and traces, all read from the v2 observations endpoint.

Everything the usage page needs lives in one place, and the two endpoints
this used to read instead are the two Langfuse rate-limits hardest:

* ``/api/public/traces`` allows **5 requests a minute**. The page made three
  of them per load, so it could not be opened twice in a minute, and the
  per-user query walked up to ten pages in a single request.
* the metrics API allows **100 requests a day**, which capped the page at
  about thirty loads before every figure read "unavailable" until morning.

Langfuse's own 429 names the way out: "Use GET /api/public/v2/observations
for high-volume reads." Measured against the live service, that endpoint
serves 1000 rows a call, pages by cursor, and carries all of it - with one
catch that shapes the whole module:

* ``user_id`` is populated **only on the root span**.
* cost and tokens are populated **only on the GENERATION children**.

So neither row type can answer alone, and filtering by ``user_id`` returns
the roots without their generations - cost 0, tokens 0. Two reads, joined
locally on ``trace_id``, is the only shape that works.

The client is stubbed: these tests are about which reads we make and what we
make of the answers. The joining, the coercion and the ordering are real.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.core import usage_stats


class FakeObservations:
    """Emulates observations.get_many: filters, cursor paging, field groups."""

    def __init__(self, rows: list[dict] | Exception, page_size: int = 1000):
        self.rows = rows
        self.page_size = page_size
        self.calls: list[dict] = []

    def get_many(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.rows, Exception):
            raise self.rows

        selected = self.rows
        if kwargs.get("is_root_observation") is not None:
            want = kwargs["is_root_observation"]
            selected = [r for r in selected if bool(r.get("is_root_observation")) is want]
        if kwargs.get("type"):
            selected = [r for r in selected if r.get("type") == kwargs["type"]]
        if kwargs.get("user_id"):
            selected = [r for r in selected if r.get("user_id") == kwargs["user_id"]]

        start = int(kwargs.get("cursor") or 0)
        limit = min(kwargs.get("limit") or self.page_size, self.page_size)
        chunk = selected[start:start + limit]
        nxt = start + limit
        # A cursor is returned only while more rows remain, which is how the
        # real endpoint signals the end of the walk.
        cursor = str(nxt) if nxt < len(selected) else None
        return SimpleNamespace(data=[SimpleNamespace(**r) for r in chunk],
                               meta=SimpleNamespace(cursor=cursor))


class FakeLangfuse:
    def __init__(self, rows: list[dict] | Exception, page_size: int = 1000):
        self.observations = FakeObservations(rows, page_size)
        self.api = self

    @property
    def calls(self) -> list[dict]:
        return self.observations.calls


def root(trace_id: str, user_id: str = "", *, question: str = "What drove revenue?",
         answer: str = "It rose 12%.", when: str = "2026-10-01T08:31:36Z",
         latency: float = 5.9, name: str = "rag-chat-streamed") -> dict:
    """A root span: carries the user, the question and the timing."""
    return {
        "id": f"obs-{trace_id}", "type": "SPAN", "is_root_observation": True,
        "trace_id": trace_id, "trace_name": name, "name": name,
        "user_id": user_id, "session_id": f"s-{trace_id}",
        "input": question, "output": answer, "tags": ["chat", "rag"],
        "start_time": when, "latency": latency,
        "total_cost": None, "usage_details": {},
    }


def generation(trace_id: str, cost: float, tokens: int) -> dict:
    """A generation: carries the money, and no user at all."""
    return {
        "id": f"gen-{trace_id}", "type": "GENERATION", "is_root_observation": False,
        "trace_id": trace_id, "trace_name": "rag-chat-streamed",
        "name": "llm-completion", "user_id": "", "session_id": "",
        "input": "", "output": "", "tags": [],
        "start_time": "2026-10-01T08:31:30Z", "latency": 4.2,
        "total_cost": cost,
        "usage_details": {"input": tokens - 20, "output": 20, "total": tokens},
    }


# ── The reads we make ────────────────────────────────────────────────

def test_cost_and_tokens_are_requested_explicitly(monkeypatch):
    """fields defaults to "core,basic", and a read without the usage and
    metrics groups comes back with total_cost=None on every row. That is how
    the first attempt at this reported no spend at all."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse([root("t1", "ada"), generation("t1", 0.004, 2420)])

    usage_stats.usage_snapshot(lf, days=30)

    for call in lf.calls:
        fields = call["fields"]
        assert "usage" in fields, "without the usage group every token count is null"
        assert "metrics" in fields, "without the metrics group every cost is null"


def test_the_window_is_sent_as_datetimes(monkeypatch):
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse([root("t1", "ada")])

    usage_stats.usage_snapshot(lf, days=7)

    call = lf.calls[0]
    assert isinstance(call["from_start_time"], datetime)
    assert isinstance(call["to_start_time"], datetime)
    assert call["from_start_time"].tzinfo is not None, "a naive datetime shifts the window"
    assert (call["to_start_time"] - call["from_start_time"]).days == 7


def test_nothing_retries_a_rate_limit(monkeypatch):
    """A 429 on the trace endpoint carries a retry-after of a minute and on
    metrics nearly a day. Retrying held a worker thread for 122 seconds."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse([root("t1", "ada")])

    usage_stats.usage_snapshot(lf, days=30)

    assert lf.calls[0]["request_options"]["max_retries"] == 0


def test_a_whole_page_costs_two_reads(monkeypatch):
    """The point of the rewrite. It used to be three trace.list calls against
    a five-a-minute limit plus three metrics calls against a hundred a day."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    rows = []
    for i in range(40):
        rows.append(root(f"t{i}", "ada"))
        rows.append(generation(f"t{i}", 0.004, 2000))
    lf = FakeLangfuse(rows)

    usage_stats.usage_snapshot(lf, days=30)

    assert len(lf.calls) == 2, [c.get("type") or "roots" for c in lf.calls]


def test_the_trace_endpoint_is_never_touched(monkeypatch):
    """It allows five requests a minute. Nothing here may reach for it."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])

    def boom(**_k):
        raise AssertionError("trace.list was called; it allows 5 a minute")

    lf = FakeLangfuse([root("t1", "ada"), generation("t1", 0.004, 2420)])
    lf.trace = SimpleNamespace(list=boom)
    lf.metrics = SimpleNamespace(metrics=lambda **_k: (_ for _ in ()).throw(
        AssertionError("the metrics API was called; it allows 100 a day")))

    usage_stats.usage_snapshot(lf, days=30)


def test_the_cursor_is_followed_to_the_end(monkeypatch):
    """1000 rows a call, and a project busier than that must not silently
    report only its first page."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    rows = [root(f"t{i}", "ada") for i in range(250)]
    lf = FakeLangfuse(rows, page_size=100)

    snapshot = usage_stats.usage_snapshot(lf, days=30)

    assert len(snapshot["traces"]) == 250
    cursors = [c.get("cursor") for c in lf.calls if c.get("is_root_observation")]
    assert cursors[0] is None and cursors[1] is not None, "the cursor was not passed back"


def test_the_walk_is_bounded_and_says_when_it_truncated(monkeypatch):
    """A busy deployment must still load, and a figure built from part of the
    data must not be presented as the whole bill."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    rows = [root(f"t{i}", "ada") for i in range(5000)]
    lf = FakeLangfuse(rows, page_size=100)

    snapshot = usage_stats.usage_snapshot(lf, days=30)

    roots_read = len([c for c in lf.calls if c.get("is_root_observation")])
    assert roots_read <= usage_stats.MAX_OBSERVATION_PAGES
    assert snapshot["truncated"] is True


def test_a_small_project_does_not_claim_truncation(monkeypatch):
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse([root("t1", "ada"), generation("t1", 0.004, 2420)])

    assert usage_stats.usage_snapshot(lf, days=30)["truncated"] is False


# ── Joining the two row types ────────────────────────────────────────

def test_a_trace_takes_its_user_from_the_root_and_its_money_from_the_child(monkeypatch):
    """Neither row can answer alone: the root has no cost, the generation has
    no user. This is the join the whole module exists for."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse([root("t1", "ada"), generation("t1", 0.00459, 2420)])

    trace = usage_stats.usage_snapshot(lf, days=30)["traces"][0]

    assert trace["user_id"] == "ada"
    assert trace["tokens"] == 2420
    assert trace["total_cost"] == pytest.approx(0.00459)


def test_a_trace_with_several_generations_sums_them(monkeypatch):
    """One question can make more than one model call - the follow-up round
    and the faithfulness gate both do."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse([
        root("t1", "ada"),
        generation("t1", 0.004, 2000),
        generation("t1", 0.001, 500),
    ])

    trace = usage_stats.usage_snapshot(lf, days=30)["traces"][0]

    assert trace["tokens"] == 2500
    assert trace["total_cost"] == pytest.approx(0.005)


def test_a_trace_that_never_reached_a_model_is_free_not_unknown(monkeypatch):
    """Small talk is answered without a model call, so it has no generation
    at all. Zero is the true answer, and must not read as missing data."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse([root("t1", "ada", question="hi", answer="Hello!")])

    trace = usage_stats.usage_snapshot(lf, days=30)["traces"][0]

    assert trace["tokens"] == 0
    assert trace["total_cost"] == 0


def test_the_trace_row_carries_what_the_table_draws(monkeypatch):
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse([root("t1", "ada", question="What drove revenue?",
                            answer="It rose 12%.", latency=5.9)])

    trace = usage_stats.usage_snapshot(lf, days=30)["traces"][0]

    assert trace["id"] == "t1"
    assert trace["input"] == "What drove revenue?"
    assert trace["output"] == "It rose 12%."
    assert trace["latency"] == pytest.approx(5.9)
    assert trace["name"] == "rag-chat-streamed"
    assert trace["tags"] == ["chat", "rag"]
    assert trace["session_id"] == "s-t1"
    assert "2026-10-01" in trace["created_at"]


def test_traces_come_back_newest_first(monkeypatch):
    """The join is by dictionary, so the order has to be restored explicitly."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse([
        root("old", "ada", when="2026-09-20T10:00:00Z"),
        root("new", "ada", when="2026-10-01T10:00:00Z"),
        root("mid", "ada", when="2026-09-25T10:00:00Z"),
    ])

    ids = [t["id"] for t in usage_stats.usage_snapshot(lf, days=30)["traces"]]

    assert ids == ["new", "mid", "old"]


# ── Per-user spend ───────────────────────────────────────────────────

def test_spend_is_attributed_to_the_user_on_the_root(monkeypatch):
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse([
        root("t1", "ada"), generation("t1", 0.01, 1000),
        root("t2", "ada"), generation("t2", 0.02, 2000),
        root("t3", "bob"), generation("t3", 0.005, 500),
    ])

    by_user = {u["user_id"]: u for u in usage_stats.usage_snapshot(lf, days=30)["users"]}

    assert by_user["ada"]["cost"] == pytest.approx(0.03)
    assert by_user["ada"]["tokens"] == 3000
    assert by_user["ada"]["traces"] == 2
    assert by_user["bob"]["cost"] == pytest.approx(0.005)


def test_users_are_joined_to_the_account_they_belong_to(monkeypatch):
    """Langfuse stores only the hex id, which an admin cannot act on."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [
        {"user_id": "abc123", "username": "ada@example.test", "role": "user"},
    ])
    lf = FakeLangfuse([root("t1", "abc123"), generation("t1", 0.07, 100)])

    snapshot = usage_stats.usage_snapshot(lf, days=30)

    assert snapshot["users"][0]["username"] == "ada@example.test"
    assert snapshot["users"][0]["role"] == "user"
    # The trace row carries it too, so the table need not resolve ids itself.
    assert snapshot["traces"][0]["username"] == "ada@example.test"


def test_an_account_deleted_since_its_traces_were_written_still_shows(monkeypatch):
    """Dropping the row would make its spend vanish from the total."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse([root("t1", "ghost"), generation("t1", 0.001, 10)])

    user = usage_stats.usage_snapshot(lf, days=30)["users"][0]

    assert user["user_id"] == "ghost"
    assert "deleted" in user["username"].lower()


def test_traces_with_no_user_are_reported_rather_than_dropped(monkeypatch):
    """Document ingestion and evaluation runs carry no user, and the vision
    preprocessing they do is real money."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse([
        root("t1", "", name="document-ingest"), generation("t1", 0.002, 300),
        root("t2", "ada"), generation("t2", 0.001, 100),
    ])

    users = usage_stats.usage_snapshot(lf, days=30)["users"]

    unattributed = next(u for u in users if u["user_id"] == "")
    assert unattributed["cost"] == pytest.approx(0.002)
    assert unattributed["username"] == usage_stats.UNATTRIBUTED


def test_users_come_back_most_expensive_first(monkeypatch):
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse([
        root("t1", "cheap"), generation("t1", 0.001, 10),
        root("t2", "dear"), generation("t2", 0.5, 10),
        root("t3", "middling"), generation("t3", 0.05, 10),
    ])

    users = usage_stats.usage_snapshot(lf, days=30)["users"]

    assert [u["user_id"] for u in users] == ["dear", "middling", "cheap"]


# ── Totals ───────────────────────────────────────────────────────────

def test_totals_are_the_sum_of_what_was_read(monkeypatch):
    """Previously a separate metrics call, and a separate trace.list call for
    the count. Both are already in hand."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse([
        root("t1", "ada"), generation("t1", 0.04, 12000),
        root("t2", "bob"), generation("t2", 0.034016, 13194),
        root("t3", "bob", question="hi"),
    ])

    totals = usage_stats.usage_snapshot(lf, days=30)["totals"]

    assert totals["cost"] == pytest.approx(0.074016)
    assert totals["tokens"] == 25194
    assert totals["traces"] == 3, "the greeting is a question that was asked"


def test_an_empty_project_reports_zeroes_rather_than_failing(monkeypatch):
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse([])

    snapshot = usage_stats.usage_snapshot(lf, days=30)

    assert snapshot["totals"] == {"cost": 0.0, "tokens": 0, "traces": 0}
    assert snapshot["traces"] == []
    assert snapshot["users"] == []


# ── Failure ──────────────────────────────────────────────────────────

def test_a_langfuse_failure_is_reported_not_swallowed(monkeypatch):
    """An empty page and a rate-limited one must not look the same."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse(RuntimeError("429 Rate limit exceeded"))

    with pytest.raises(usage_stats.UsageUnavailable, match="429"):
        usage_stats.usage_snapshot(lf, days=30)


def test_a_cost_that_arrives_as_a_string_still_adds_up(monkeypatch):
    """The metrics API quoted its numbers; this one is not trusted either."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    gen = generation("t1", 0.004, 2420)
    gen["total_cost"] = "0.004"
    gen["usage_details"] = {"total": "2420"}
    lf = FakeLangfuse([root("t1", "ada"), gen])

    trace = usage_stats.usage_snapshot(lf, days=30)["traces"][0]

    assert trace["total_cost"] == pytest.approx(0.004)
    assert trace["tokens"] == 2420
    assert isinstance(trace["tokens"], int)


def test_a_generation_with_no_usage_block_counts_as_nothing(monkeypatch):
    """usage_details comes back {} on spans and can be absent on a failed
    generation. That must not raise."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    gen = generation("t1", 0.004, 2420)
    gen["usage_details"] = None
    lf = FakeLangfuse([root("t1", "ada"), gen])

    trace = usage_stats.usage_snapshot(lf, days=30)["traces"][0]

    assert trace["tokens"] == 0
    assert trace["total_cost"] == pytest.approx(0.004)


def test_a_generation_whose_trace_was_not_read_does_not_invent_a_row(monkeypatch):
    """When the root walk truncates, generations can arrive for traces that
    are not on the list. Those must not become rows with no question."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse([root("t1", "ada"), generation("t1", 0.004, 100),
                       generation("orphan", 0.009, 900)])

    snapshot = usage_stats.usage_snapshot(lf, days=30)

    assert [t["id"] for t in snapshot["traces"]] == ["t1"]
    # The spend is real, so it still belongs in the total.
    assert snapshot["totals"]["cost"] == pytest.approx(0.013)
