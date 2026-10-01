"""Per-user spend and token totals, read from Langfuse's metrics API.

The Langfuse client is stubbed: these tests are about the query we send and
what we make of the answer, and the real endpoint is a network call to a
hosted service. Everything else - the username join, the string-to-number
coercion, the ordering - is real code.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from app.core import usage_stats


class FakeMetrics:
    """Records the query it was asked, returns whatever it was given."""

    def __init__(self, rows: list[dict] | Exception):
        self.rows = rows
        self.queries: list[dict] = []
        self.options: list[dict | None] = []

    def metrics(self, *, query: str, request_options=None):
        self.queries.append(json.loads(query))
        self.options.append(request_options)
        if isinstance(self.rows, Exception):
            raise self.rows

        class Response:
            data = self.rows

        return Response()


class FakeTraceList:
    """Pages of traces, the way lf.api.trace.list serves them."""

    def __init__(self, traces: list[dict] | Exception, page_size: int = 100):
        self.traces = traces
        self.page_size = page_size
        self.calls: list[dict] = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.traces, Exception):
            raise self.traces
        page = kwargs.get("page", 1)
        start = (page - 1) * self.page_size
        chunk = self.traces[start:start + self.page_size]
        total_pages = max(1, -(-len(self.traces) // self.page_size))
        return SimpleNamespace(
            data=[SimpleNamespace(**t) for t in chunk],
            meta=SimpleNamespace(total_items=len(self.traces), total_pages=total_pages),
        )


class FakeLangfuse:
    def __init__(self, rows: list[dict] | Exception, traces: list[dict] | Exception | None = None,
                 page_size: int = 100):
        self.metrics = FakeMetrics(rows)
        self.trace = FakeTraceList(traces if traces is not None else [], page_size)
        self.api = self

    @property
    def last_query(self) -> dict:
        return self.metrics.queries[-1]


def _trace(id_: str, user_id: str, cost: float) -> dict:
    return {"id": id_, "user_id": user_id, "total_cost": cost}


# ── Where the per-user numbers come from ─────────────────────────────
#
# Not from grouping observations by userId, which is the obvious reading of
# the metrics API and is wrong: the SDK does not carry a trace's user down to
# the generations beneath it, so every observation has user_id="" and the
# whole bill lands in one unattributed row. Traces do carry the user and the
# cost, and observations carry the tokens, so the two are joined on traceId.

def test_spend_per_user_is_summed_from_traces_not_observations(monkeypatch):
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse(
        rows=[],
        traces=[_trace("t1", "ada", 0.01), _trace("t2", "ada", 0.02), _trace("t3", "bob", 0.005)],
    )

    rows = usage_stats.per_user_usage(lf, days=30)

    by_user = {r["user_id"]: r for r in rows}
    assert by_user["ada"]["cost"] == pytest.approx(0.03)
    assert by_user["bob"]["cost"] == pytest.approx(0.005)


def test_each_user_reports_how_many_questions_they_asked(monkeypatch):
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse(rows=[], traces=[
        _trace("t1", "ada", 0.01), _trace("t2", "ada", 0.02), _trace("t3", "bob", 0.005),
    ])

    by_user = {r["user_id"]: r for r in usage_stats.per_user_usage(lf, days=30)}

    assert by_user["ada"]["traces"] == 2
    assert by_user["bob"]["traces"] == 1


def test_tokens_are_attributed_by_joining_observations_to_their_trace(monkeypatch):
    """Tokens exist only per trace, and users exist only per trace, so the
    trace id is the only thing that can connect them."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse(
        rows=[
            {"traceId": "t1", "sum_totalTokens": "2000"},
            {"traceId": "t2", "sum_totalTokens": "400"},
            {"traceId": "t3", "sum_totalTokens": "111"},
        ],
        traces=[_trace("t1", "ada", 0.01), _trace("t2", "ada", 0.02), _trace("t3", "bob", 0.005)],
    )

    by_user = {r["user_id"]: r for r in usage_stats.per_user_usage(lf, days=30)}

    assert by_user["ada"]["tokens"] == 2400
    assert by_user["bob"]["tokens"] == 111


def test_the_token_query_groups_by_trace_and_satisfies_the_cardinality_rule(monkeypatch):
    """traceId is high-cardinality too: Langfuse 400s without both a row
    limit and an orderBy on a measure."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse(rows=[], traces=[_trace("t1", "ada", 0.01)])

    usage_stats.per_user_usage(lf, days=30)

    q = lf.last_query
    assert q["view"] == "observations"
    assert q["dimensions"] == [{"field": "traceId"}]
    assert q["config"]["row_limit"] >= 1
    assert q["orderBy"][0]["direction"] == "desc"
    assert q["orderBy"][0]["field"].startswith("sum_")


def test_the_row_limit_stays_inside_what_langfuse_accepts(monkeypatch):
    """row_limit has a ceiling. Asking for 5000 is rejected as "Invalid
    request data", and because the token query fails soft, the only symptom
    was every user reporting 0 tokens while their costs were right."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse(rows=[], traces=[_trace("t1", "ada", 0.01)])

    usage_stats.per_user_usage(lf, days=30)

    assert lf.last_query["config"]["row_limit"] <= usage_stats.MAX_ROW_LIMIT
    assert usage_stats.MAX_ROW_LIMIT <= 1000


def test_no_more_traces_are_read_than_tokens_can_be_fetched_for(monkeypatch):
    """Walking 2,000 traces when only 1,000 can carry a token count leaves a
    tail of rows whose tokens read 0 for no visible reason."""
    assert usage_stats.MAX_TRACE_PAGES * 100 <= usage_stats.MAX_ROW_LIMIT


def test_a_user_whose_tokens_langfuse_did_not_return_still_shows_their_spend(monkeypatch):
    """The token query is capped by row_limit, so a long tail has no tokens.
    Hiding those users would hide real spend."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse(rows=[], traces=[_trace("t1", "ada", 0.01)])

    rows = usage_stats.per_user_usage(lf, days=30)

    assert rows[0]["cost"] == pytest.approx(0.01)
    assert rows[0]["tokens"] == 0


def test_rows_are_joined_to_the_account_they_belong_to(monkeypatch):
    """Langfuse only stores the hex user id. An admin cannot act on that."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [
        {"user_id": "abc123", "username": "ada@example.test", "role": "user"},
        {"user_id": "def456", "username": "grace@example.test", "role": "admin"},
    ])
    lf = FakeLangfuse(rows=[], traces=[_trace("t1", "abc123", 0.07), _trace("t2", "def456", 0.003)])

    rows = usage_stats.per_user_usage(lf, days=30)

    assert rows[0]["username"] == "ada@example.test"
    assert rows[0]["role"] == "user"
    assert rows[1]["username"] == "grace@example.test"
    assert rows[1]["role"] == "admin"


def test_token_counts_arrive_as_strings_and_come_back_as_numbers(monkeypatch):
    """The API really does return "25194" quoted; summing those concatenates."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse(rows=[{"traceId": "t1", "sum_totalTokens": "25194"}],
                      traces=[_trace("t1", "u1", 0.074)])

    rows = usage_stats.per_user_usage(lf, days=30)

    assert rows[0]["tokens"] == 25194
    assert isinstance(rows[0]["tokens"], int)
    assert rows[0]["cost"] == pytest.approx(0.074)


def test_an_account_deleted_since_its_traces_were_written_still_shows(monkeypatch):
    """Dropping the row would make the spend vanish from the totals."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse(rows=[], traces=[_trace("t1", "ghost", 0.001)])

    rows = usage_stats.per_user_usage(lf, days=30)

    assert len(rows) == 1
    assert rows[0]["user_id"] == "ghost"
    assert "deleted" in rows[0]["username"].lower()


def test_traces_with_no_user_are_reported_rather_than_dropped(monkeypatch):
    """Document ingestion and evaluation runs carry no user, and their
    spend is real money."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse(rows=[], traces=[_trace("t1", "", 0.002), _trace("t2", "u1", 0.001)])

    rows = usage_stats.per_user_usage(lf, days=30)

    assert len(rows) == 2
    unattributed = next(r for r in rows if r["user_id"] == "")
    assert unattributed["cost"] == pytest.approx(0.002)
    assert unattributed["username"] == usage_stats.UNATTRIBUTED


def test_rows_come_back_most_expensive_first(monkeypatch):
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse(rows=[], traces=[
        _trace("t1", "cheap", 0.001), _trace("t2", "dear", 0.5), _trace("t3", "middling", 0.05),
    ])

    rows = usage_stats.per_user_usage(lf, days=30)

    assert [r["user_id"] for r in rows] == ["dear", "middling", "cheap"]


def test_the_trace_window_is_sent_to_langfuse_as_datetimes(monkeypatch):
    """Without a window, "last 30 days" would quietly mean "all time".

    And it must be datetime objects: the trace client calls .tzinfo on what
    it is given, so an ISO string - which is what the metrics API wants -
    fails with "'str' object has no attribute 'tzinfo'". A stub that accepts
    anything hides this, so the type is asserted here.
    """
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse(rows=[], traces=[_trace("t1", "u1", 0.01)])

    usage_stats.per_user_usage(lf, days=7)

    call = lf.trace.calls[0]
    assert isinstance(call.get("from_timestamp"), datetime)
    assert isinstance(call.get("to_timestamp"), datetime)
    assert call["from_timestamp"].tzinfo is not None, "a naive datetime shifts the window"
    assert (call["to_timestamp"] - call["from_timestamp"]).days == 7


def test_the_metrics_window_is_sent_as_iso_strings(monkeypatch):
    """The other half of the same trap: the metrics API takes strings."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse(rows=[], traces=[_trace("t1", "u1", 0.01)])

    usage_stats.per_user_usage(lf, days=7)

    assert isinstance(lf.last_query["fromTimestamp"], str)
    assert isinstance(lf.last_query["toTimestamp"], str)


def test_paging_stops_at_the_cap_rather_than_walking_forever(monkeypatch):
    """A busy deployment has thousands of traces; the page must still load."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    many = [_trace(f"t{i}", f"u{i % 3}", 0.001) for i in range(1000)]
    lf = FakeLangfuse(rows=[], traces=many, page_size=100)

    rows = usage_stats.per_user_usage(lf, days=30)

    assert len(lf.trace.calls) <= usage_stats.MAX_TRACE_PAGES
    assert rows, "the cap swallowed every row"


def test_a_langfuse_failure_is_reported_not_swallowed(monkeypatch):
    """An empty table and a broken connection must not look the same."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse(rows=[], traces=RuntimeError("401 unauthorized"))

    with pytest.raises(usage_stats.UsageUnavailable, match="401"):
        usage_stats.per_user_usage(lf, days=30)


def test_tokens_failing_does_not_lose_the_spend(monkeypatch):
    """Cost is the number that matters; a token query that errors must not
    take the whole table down with it."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse(rows=RuntimeError("metrics unavailable"),
                      traces=[_trace("t1", "ada", 0.01)])

    rows = usage_stats.per_user_usage(lf, days=30)

    assert rows[0]["cost"] == pytest.approx(0.01)
    assert rows[0]["tokens"] == 0


# ── Totals, which is what the broken stat card needs ─────────────────

def test_totals_read_the_aggregate_row():
    lf = FakeLangfuse([{"sum_totalTokens": "25194", "sum_totalCost": 0.07401625}])

    totals = usage_stats.totals(lf, days=30)

    assert totals["tokens"] == 25194
    assert totals["cost"] == pytest.approx(0.07401625)


def test_totals_of_an_empty_project_are_zero_not_an_error():
    lf = FakeLangfuse([])

    totals = usage_stats.totals(lf, days=30)

    assert totals == {"tokens": 0, "cost": 0.0}


def test_the_totals_query_asks_for_no_dimension():
    """A dimension would return one row per group and break the sum."""
    lf = FakeLangfuse([])
    usage_stats.totals(lf, days=30)

    assert lf.last_query["dimensions"] == []
    assert lf.last_query["view"] == "observations"


# ── Tokens for a page of traces ──────────────────────────────────────
#
# One grouped query for the whole page, not one query per trace. Langfuse
# allows 100 API requests a day: a query per expanded row let a single
# afternoon of clicking exhaust the quota for everybody, after which every
# figure on the page read "unavailable" until the next day.

def test_tokens_are_returned_for_every_trace_on_the_page():
    lf = FakeLangfuse([
        {"traceId": "t1", "sum_totalTokens": "1843"},
        {"traceId": "t2", "sum_totalTokens": "2420"},
    ])

    tokens = usage_stats.tokens_for_traces(lf, ["t1", "t2"], days=30)

    assert tokens == {"t1": 1843, "t2": 2420}


def test_a_whole_page_of_traces_costs_exactly_one_request():
    """The quota is the reason this function exists at all."""
    lf = FakeLangfuse([{"traceId": f"t{i}", "sum_totalTokens": "10"} for i in range(20)])

    usage_stats.tokens_for_traces(lf, [f"t{i}" for i in range(20)], days=30)

    assert len(lf.metrics.queries) == 1


def test_asking_about_no_traces_asks_langfuse_nothing():
    """An empty table must not spend a request to learn it is empty."""
    lf = FakeLangfuse([])

    assert usage_stats.tokens_for_traces(lf, [], days=30) == {}
    assert lf.metrics.queries == []


def test_a_trace_that_spent_nothing_reports_zero():
    """A greeting is answered without a model call, so it produces no
    observations at all and is simply absent from the grouping. That is a
    real answer - the question was free - not a missing one."""
    lf = FakeLangfuse([{"traceId": "t1", "sum_totalTokens": "1843"}])

    tokens = usage_stats.tokens_for_traces(lf, ["t1", "greeting"], days=30)

    assert tokens["greeting"] == 0


def test_a_trace_beyond_the_row_limit_reports_unknown_rather_than_zero():
    """Absence means two different things. Under the row limit the grouping
    is complete, so a missing trace really did spend nothing. At the limit
    the list was truncated, and calling that tail "free" understates the
    bill in exactly the place an admin is looking for overspend.
    """
    full = [{"traceId": f"t{i}", "sum_totalTokens": "10"}
            for i in range(usage_stats.MAX_ROW_LIMIT)]
    lf = FakeLangfuse(full)

    tokens = usage_stats.tokens_for_traces(lf, ["t0", "not-in-the-page"], days=30)

    assert tokens["t0"] == 10
    assert tokens["not-in-the-page"] is None, "a truncated grouping claimed the trace was free"


def test_tokens_for_a_page_survive_a_langfuse_failure():
    """The count is a nicety; it must not stop the trace table rendering."""
    lf = FakeLangfuse(RuntimeError("429 Rate limit exceeded"))

    tokens = usage_stats.tokens_for_traces(lf, ["t1", "t2"], days=30)

    assert tokens == {"t1": None, "t2": None}, "a rate limit was drawn as a free answer"


# ── Failing fast ─────────────────────────────────────────────────────
#
# When the daily quota is spent, Langfuse answers 429 with a retry-after of
# the better part of a day. The SDK's default is to retry with backoff, so
# the call sat there for 122 seconds before giving up - holding a worker
# thread, with the page showing nothing. Retrying cannot help when the
# window reopens tomorrow morning.

def test_a_metrics_query_does_not_retry():
    lf = FakeLangfuse([])

    usage_stats.tokens_for_traces(lf, ["t1"], days=30)

    opts = lf.metrics.options[-1]
    assert opts is not None, "the SDK default of retrying with backoff still applies"
    assert opts["max_retries"] == 0
    assert 0 < opts["timeout_in_seconds"] <= 30, "an unbounded wait holds a worker thread"


def test_the_trace_list_does_not_retry_either(monkeypatch):
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])
    lf = FakeLangfuse(rows=[], traces=[_trace("t1", "u1", 0.01)])

    usage_stats.per_user_usage(lf, days=30)

    opts = lf.trace.calls[0].get("request_options")
    assert opts is not None, "a rate-limited trace page retried for two minutes"
    assert opts["max_retries"] == 0


def test_the_page_token_query_satisfies_the_cardinality_rule():
    """traceId is high-cardinality: Langfuse 400s without both a row limit
    and an orderBy on a measure."""
    lf = FakeLangfuse([])

    usage_stats.tokens_for_traces(lf, ["t1"], days=30)

    q = lf.last_query
    assert q["view"] == "observations"
    assert q["dimensions"] == [{"field": "traceId"}]
    assert q["config"]["row_limit"] <= usage_stats.MAX_ROW_LIMIT
    assert q["orderBy"][0]["direction"] == "desc"
