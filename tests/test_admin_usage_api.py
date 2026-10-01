"""The admin usage endpoint: one request, one window, everything on it.

Four routes became one. Each of the old four made its own Langfuse calls,
and together they spent three requests against ``/api/public/traces`` - a
five-a-minute limit - plus three against the metrics API, which allows a
hundred a day. The page could not be opened twice in a minute.

Langfuse is stubbed at the one seam the route uses, ``_get_langfuse``. The
routing, the admin guard and the shape of the reply are all real.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

import app.api.routes.admin as admin_routes
from app.core import usage_stats


class StubObservations:
    """Returns roots or generations depending on the filter it is given."""

    def __init__(self):
        self.calls: list[dict] = []
        self.roots = [SimpleNamespace(
            id="obs1", type="SPAN", is_root_observation=True,
            trace_id="t1", trace_name="rag-chat-streamed", name="rag-chat-streamed",
            user_id="u1", session_id="s1",
            input="What drove revenue?", output="It rose 12%.",
            tags=["chat", "rag"], start_time="2026-10-01T08:31:36Z", latency=5.9,
            total_cost=None, usage_details={},
        )]
        self.generations = [SimpleNamespace(
            id="gen1", type="GENERATION", is_root_observation=False,
            trace_id="t1", trace_name="rag-chat-streamed", name="llm-completion",
            user_id="", session_id="",
            input="", output="", tags=[],
            start_time="2026-10-01T08:31:30Z", latency=4.2,
            total_cost=0.0042, usage_details={"input": 2391, "output": 29, "total": 2420},
        )]

    def get_many(self, **kwargs):
        self.calls.append(kwargs)
        data = self.generations if kwargs.get("type") == "GENERATION" else self.roots
        return SimpleNamespace(data=data, meta=SimpleNamespace(cursor=None))


class StubLangfuse:
    def __init__(self):
        self.observations = StubObservations()
        self.api = self


@pytest.fixture
def lf(monkeypatch):
    stub = StubLangfuse()
    monkeypatch.setattr(admin_routes, "_get_langfuse", lambda: stub)
    monkeypatch.setattr(usage_stats, "list_all_users", lambda: [
        {"user_id": "u1", "username": "ada@example.test", "role": "user"},
    ])
    return stub


@pytest.fixture
def admin_client(client):
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


# ── One request serves the page ──────────────────────────────────────

def test_the_whole_page_arrives_in_one_response(admin_client, lf):
    resp = admin_client.get("/api/v1/admin/langfuse/usage?days=30")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["enabled"] is True
    assert body["traces"][0]["id"] == "t1"
    assert body["users"][0]["username"] == "ada@example.test"
    assert body["totals"]["tokens"] == 2420
    assert body["totals"]["cost"] == pytest.approx(0.0042)
    assert body["truncated"] is False


def test_it_spends_two_langfuse_requests(admin_client, lf):
    """The reason the four routes became one."""
    admin_client.get("/api/v1/admin/langfuse/usage")

    assert len(lf.observations.calls) == 2


def test_a_trace_row_carries_its_user_and_its_tokens(admin_client, lf):
    """The user is on the root span, the tokens on the generation beneath it,
    and the row needs both."""
    row = admin_client.get("/api/v1/admin/langfuse/usage").json()["traces"][0]

    assert row["user_id"] == "u1"
    assert row["username"] == "ada@example.test"
    assert row["tokens"] == 2420
    assert row["total_cost"] == pytest.approx(0.0042)
    assert row["latency"] == pytest.approx(5.9)


def test_the_window_is_honoured(admin_client, lf):
    admin_client.get("/api/v1/admin/langfuse/usage?days=7")

    call = lf.observations.calls[0]
    assert (call["to_start_time"] - call["from_start_time"]).days == 7


# ── The endpoints that cost too much are gone ────────────────────────

@pytest.mark.parametrize("path", [
    "/api/v1/admin/langfuse/traces",
    "/api/v1/admin/langfuse/by-user",
    "/api/v1/admin/langfuse/summary",
    "/api/v1/admin/langfuse/trace-tokens/t1",
])
def test_the_routes_that_read_the_rate_limited_endpoints_are_gone(admin_client, lf, path):
    """Left in place they would still be reachable, and each one still spent
    a request against a five-a-minute limit."""
    assert admin_client.get(path).status_code == 404, f"{path} still exists"


def test_neither_rate_limited_endpoint_is_touched(admin_client, lf):
    lf.trace = SimpleNamespace(list=lambda **_k: (_ for _ in ()).throw(
        AssertionError("trace.list was called; it allows 5 a minute")))
    lf.metrics = SimpleNamespace(metrics=lambda **_k: (_ for _ in ()).throw(
        AssertionError("the metrics API was called; it allows 100 a day")))

    assert admin_client.get("/api/v1/admin/langfuse/usage").status_code == 200


# ── Degrading honestly ───────────────────────────────────────────────

def test_it_says_when_langfuse_is_not_configured(admin_client, monkeypatch):
    monkeypatch.setattr(admin_routes, "_get_langfuse", lambda: None)

    body = admin_client.get("/api/v1/admin/langfuse/usage").json()

    assert body["enabled"] is False


def test_a_rate_limit_is_reported_rather_than_shown_as_no_spend(admin_client, lf):
    """"Nobody has spent anything" and "we were rate-limited" are different
    answers, and the page spent a day drawing the second as the first."""
    def boom(**_k):
        raise RuntimeError("429 Rate limit exceeded")

    lf.observations.get_many = boom

    resp = admin_client.get("/api/v1/admin/langfuse/usage")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "429" in body["error"]
    assert body["traces"] == []
    assert body["users"] == []


def test_truncation_is_passed_on(admin_client, lf, monkeypatch):
    """A bill built from part of the data must not look like all of it."""
    monkeypatch.setattr(usage_stats, "MAX_OBSERVATION_PAGES", 1)

    def always_more(**kwargs):
        lf.observations.calls.append(kwargs)
        data = lf.observations.generations if kwargs.get("type") == "GENERATION" else lf.observations.roots
        return SimpleNamespace(data=data, meta=SimpleNamespace(cursor="more"))

    lf.observations.get_many = always_more

    assert admin_client.get("/api/v1/admin/langfuse/usage").json()["truncated"] is True


# ── Not on the event loop ────────────────────────────────────────────

def test_the_usage_route_does_not_run_on_the_event_loop():
    """The Langfuse SDK is synchronous. Called from an `async def` route it
    blocks uvicorn's single event loop for the whole round trip, which wedged
    the entire API - sign-in included - while the usage page loaded.

    A plain `def` route is run in FastAPI's threadpool instead. This asserts
    the shape rather than the timing, which only shows up under concurrency
    against a live service.
    """
    import inspect

    assert not inspect.iscoroutinefunction(admin_routes.langfuse_usage), (
        "langfuse_usage calls a synchronous SDK from the event loop; "
        "declare it `def` so FastAPI runs it in a worker thread"
    )


# ── The guard ────────────────────────────────────────────────────────

def test_a_normal_user_cannot_read_what_everyone_spends(client, lf):
    from app.core.user_store import register_user

    username = f"nosy_{uuid.uuid4().hex[:8]}@example.test"
    register_user(username, "correct-horse-battery")
    client.cookies.clear()
    client.post("/api/v1/auth/login",
                json={"username": username, "password": "correct-horse-battery"})

    assert client.get("/api/v1/admin/langfuse/usage").status_code == 403
    client.cookies.clear()
