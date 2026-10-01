"""The admin usage endpoints: per-user spend, and filtering the trace list.

Langfuse is stubbed at the one seam the routes use, ``_get_langfuse``. The
routing, the admin guard, the parameter handling and the shape of the reply
are all real.
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest

import app.api.routes.admin as admin_routes
from app.core import usage_stats


class StubTrace:
    """Records the filters it was called with; returns one trace."""

    def __init__(self):
        self.calls: list[dict] = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(
                id="t1", name="rag-chat-streamed", user_id="u1", session_id="s1",
                input="What drove revenue?", output="It rose 12%.",
                tags=["chat", "rag"], timestamp="2026-10-01T08:00:00Z",
                total_cost=0.0042, latency=5.9,
            )],
            meta=SimpleNamespace(total_items=39),
        )


class StubMetrics:
    def __init__(self, rows):
        self.rows = rows
        self.queries: list[dict] = []
        self.options: list[dict | None] = []

    def metrics(self, *, query: str, request_options=None):
        self.queries.append(json.loads(query))
        self.options.append(request_options)
        return SimpleNamespace(data=self.rows)


class StubLangfuse:
    def __init__(self, rows=None):
        self.trace = StubTrace()
        self.metrics = StubMetrics(rows if rows is not None else [])
        self.api = self


@pytest.fixture
def lf(monkeypatch):
    stub = StubLangfuse()
    monkeypatch.setattr(admin_routes, "_get_langfuse", lambda: stub)
    return stub


@pytest.fixture
def admin_client(client, monkeypatch):
    """A client signed in as an admin."""
    from app.core.user_store import admin_create_user

    username = f"usageadmin_{uuid.uuid4().hex[:8]}@example.test"
    admin_create_user(username, "correct-horse-battery", role="admin")
    client.cookies.clear()
    resp = client.post("/api/v1/auth/login",
                       json={"username": username, "password": "correct-horse-battery"})
    assert resp.status_code == 200, resp.text
    yield client
    client.cookies.clear()


# ── Filtering the trace list ─────────────────────────────────────────

def test_a_user_filter_is_pushed_down_to_langfuse(admin_client, lf):
    """Filtering in the browser would only filter the 20 rows on screen."""
    resp = admin_client.get("/api/v1/admin/langfuse/traces?user_id=u1&page=1&limit=20")

    assert resp.status_code == 200, resp.text
    assert lf.trace.calls[-1]["user_id"] == "u1"


def test_sorting_is_pushed_down_too(admin_client, lf):
    resp = admin_client.get("/api/v1/admin/langfuse/traces?order_by=timestamp.desc")

    assert resp.status_code == 200
    assert lf.trace.calls[-1]["order_by"] == "timestamp.desc"


@pytest.mark.parametrize("order_by", ["totalCost.desc", "latency.desc", "nonsense", "timestamp.sideways"])
def test_an_unsupported_sort_is_ignored_rather_than_emptying_the_table(admin_client, lf, order_by):
    """Langfuse accepts only timestamp and name: anything else comes back
    "Invalid order by column", and because the route reports a failure as an
    empty list, one bad sort blanked the whole table.
    """
    resp = admin_client.get(f"/api/v1/admin/langfuse/traces?order_by={order_by}")

    assert resp.status_code == 200, resp.text
    assert resp.json()["traces"], "an unsupported sort emptied the table"
    assert lf.trace.calls[-1].get("order_by") is None, (
        f"{order_by!r} was forwarded to Langfuse, which rejects it"
    )


@pytest.mark.parametrize("order_by", ["timestamp.desc", "timestamp.asc", "name.asc", "name.desc"])
def test_the_sorts_langfuse_does_support_are_forwarded(admin_client, lf, order_by):
    admin_client.get(f"/api/v1/admin/langfuse/traces?order_by={order_by}")

    assert lf.trace.calls[-1]["order_by"] == order_by


def test_a_name_filter_is_pushed_down(admin_client, lf):
    admin_client.get("/api/v1/admin/langfuse/traces?name=rag-chat-streamed")

    assert lf.trace.calls[-1]["name"] == "rag-chat-streamed"


def test_filters_left_out_are_not_sent_as_empty_strings(admin_client, lf):
    """Langfuse treats user_id="" as "a user whose id is empty", which
    matches nothing, so an unfiltered page would come back blank."""
    admin_client.get("/api/v1/admin/langfuse/traces?page=1&limit=20")

    call = lf.trace.calls[-1]
    assert call.get("user_id") is None
    assert call.get("name") is None


def test_the_trace_row_still_carries_what_the_table_draws(admin_client, lf):
    resp = admin_client.get("/api/v1/admin/langfuse/traces")

    row = resp.json()["traces"][0]
    assert row["id"] == "t1"
    assert row["user_id"] == "u1"
    assert row["total_cost"] == pytest.approx(0.0042)
    assert row["latency"] == pytest.approx(5.9)
    assert resp.json()["total"] == 39


def test_the_trace_row_no_longer_advertises_a_token_count_it_cannot_know(admin_client, lf):
    """A trace carries no usage in SDK 4.x. The old code read it anyway and
    reported 0 for every answer, which read as "this answer was free"."""
    row = admin_client.get("/api/v1/admin/langfuse/traces").json()["traces"][0]

    assert "usage" not in row, "a token count that is always zero is worse than none"


# ── Tokens travel with the page, not one request per row ─────────────

def test_every_trace_row_carries_its_token_count(admin_client, lf):
    """Tokens used to be fetched per row, on expanding it. Langfuse allows
    100 requests a day, so clicking through two pages spent half the quota
    and the whole page then read "unavailable" until the next morning.
    """
    lf.metrics.rows = [{"traceId": "t1", "sum_totalTokens": "2420"}]

    row = admin_client.get("/api/v1/admin/langfuse/traces").json()["traces"][0]

    assert row["tokens"] == 2420


def test_the_page_spends_one_metrics_request_however_many_rows(admin_client, lf):
    lf.metrics.rows = [{"traceId": f"t{i}", "sum_totalTokens": "10"} for i in range(20)]

    admin_client.get("/api/v1/admin/langfuse/traces?limit=20")

    assert len(lf.metrics.queries) == 1


def test_a_token_query_that_fails_still_serves_the_traces(admin_client, lf):
    """A rate limit must cost the token column, not the table."""
    def boom(*_a, **_k):
        raise RuntimeError("429 Rate limit exceeded")

    lf.metrics.metrics = boom

    body = admin_client.get("/api/v1/admin/langfuse/traces").json()

    assert body["traces"], "losing the token count emptied the table"
    assert body["traces"][0]["tokens"] is None, "a rate limit was drawn as a free answer"


def test_a_trace_that_really_spent_nothing_reports_zero_not_unknown(admin_client, lf):
    """Small talk is answered without a model call. "0 tokens" is the true
    answer there, and must stay distinguishable from "we could not say"."""
    lf.metrics.rows = [{"traceId": "some-other-trace", "sum_totalTokens": "500"}]

    row = admin_client.get("/api/v1/admin/langfuse/traces").json()["traces"][0]

    assert row["tokens"] == 0


@pytest.mark.parametrize("path", [
    "/api/v1/admin/langfuse/traces",
    "/api/v1/admin/langfuse/summary",
])
def test_the_routes_own_trace_calls_do_not_retry_a_rate_limit(admin_client, lf, path):
    """The SDK retries with backoff by default. Langfuse answers 429 with a
    retry-after of nearly a day, so the retry cannot succeed - it only held
    the connection for two minutes with nothing on screen."""
    admin_client.get(path)

    opts = lf.trace.calls[-1].get("request_options")
    assert opts is not None, f"{path} still retries for two minutes"
    assert opts["max_retries"] == 0


def test_the_per_row_token_endpoint_is_gone(admin_client, lf):
    """Kept alongside the new column it would still be reachable, and still
    be one request per click against a hundred-a-day quota."""
    resp = admin_client.get("/api/v1/admin/langfuse/trace-tokens/t1")

    assert resp.status_code == 404, "the per-row endpoint still burns the quota"


# ── Per-user spend ───────────────────────────────────────────────────

def test_per_user_spend_is_served_with_real_usernames(admin_client, lf, monkeypatch):
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [
        {"user_id": "u1", "username": "ada@example.test", "role": "user"},
    ])
    # Spend comes from the trace (the stub's one trace costs 0.0042 and
    # belongs to u1); tokens come from the per-trace metrics query.
    lf.metrics.rows = [{"traceId": "t1", "sum_totalTokens": "2500"}]

    resp = admin_client.get("/api/v1/admin/langfuse/by-user")

    assert resp.status_code == 200, resp.text
    rows = resp.json()["users"]
    assert rows[0]["username"] == "ada@example.test"
    assert rows[0]["tokens"] == 2500
    assert rows[0]["cost"] == pytest.approx(0.0042)
    assert rows[0]["traces"] == 1


def test_per_user_spend_says_when_langfuse_is_not_configured(admin_client, monkeypatch):
    monkeypatch.setattr(admin_routes, "_get_langfuse", lambda: None)

    resp = admin_client.get("/api/v1/admin/langfuse/by-user")

    assert resp.status_code == 200
    assert resp.json()["enabled"] is False


def test_per_user_spend_reports_a_langfuse_error_rather_than_an_empty_table(admin_client, lf):
    """Spend is read from the traces, so losing those loses the answer."""
    def boom(*_a, **_k):
        raise RuntimeError("401 unauthorized")

    lf.trace.list = boom

    resp = admin_client.get("/api/v1/admin/langfuse/by-user")

    assert resp.status_code == 200
    assert "401" in resp.json()["error"]
    assert resp.json()["users"] == []


def test_losing_only_the_token_query_still_reports_the_spend(admin_client, lf, monkeypatch):
    """Cost is the number being asked about; tokens are the nicety."""
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [])

    def boom(*_a, **_k):
        raise RuntimeError("metrics unavailable")

    lf.metrics.metrics = boom

    body = admin_client.get("/api/v1/admin/langfuse/by-user").json()

    assert "error" not in body
    assert body["users"][0]["cost"] == pytest.approx(0.0042)
    assert body["users"][0]["tokens"] == 0


def test_the_summary_reports_tokens_from_the_observations_view(admin_client, lf):
    """The old summary read trace.usage and always said 0."""
    lf.metrics.rows = [{"sum_totalTokens": "25194", "sum_totalCost": 0.074}]

    resp = admin_client.get("/api/v1/admin/langfuse/summary")

    body = resp.json()
    assert body["total_tokens"] == 25194, "the tokens card is still reading zero"
    assert body["total_cost"] == pytest.approx(0.074)
    assert lf.metrics.queries[-1]["view"] == "observations"


# ── Not on the event loop ────────────────────────────────────────────

def test_the_langfuse_routes_do_not_run_on_the_event_loop():
    """The Langfuse SDK is synchronous. Called from an `async def` route it
    blocks uvicorn's single event loop for the whole round trip, and the
    per-user query makes several in a row - which wedged the entire API,
    sign-in included, while the usage page loaded.

    A plain `def` route is run in FastAPI's threadpool instead, so the loop
    stays free. This asserts the shape rather than the timing because the
    timing only shows up under concurrency against a live service.
    """
    import inspect

    blocking = ["langfuse_traces", "langfuse_by_user", "langfuse_summary"]
    offenders = [
        name for name in blocking
        if inspect.iscoroutinefunction(getattr(admin_routes, name))
    ]
    assert not offenders, (
        f"{offenders} call a synchronous SDK from the event loop; "
        "declare them `def` so FastAPI runs them in a worker thread"
    )


# ── The guard ────────────────────────────────────────────────────────

def test_a_normal_user_cannot_read_what_everyone_spends(client, lf):
    from app.core.user_store import register_user

    username = f"nosy_{uuid.uuid4().hex[:8]}@example.test"
    register_user(username, "correct-horse-battery")
    client.cookies.clear()
    client.post("/api/v1/auth/login",
                json={"username": username, "password": "correct-horse-battery"})

    for path in ("/api/v1/admin/langfuse/by-user", "/api/v1/admin/langfuse/traces"):
        assert client.get(path).status_code == 403, path
    client.cookies.clear()
