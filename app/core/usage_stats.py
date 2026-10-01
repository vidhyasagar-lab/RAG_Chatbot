"""Spend and token totals from Langfuse's metrics API.

Three facts about that API shape everything here, each of them learned the
hard way against the live service:

* **Tokens are not on a trace.** A trace object in SDK 4.x has no ``usage``
  attribute at all, which is why the usage page reported 0 tokens for every
  answer while costs were correct. Tokens live on the generations beneath a
  trace, so every count here comes from the ``observations`` view.
* **``userId`` is high cardinality.** Grouping by it is refused outright
  unless the query also carries ``config.row_limit`` and an ``orderBy`` on a
  measure, descending.
* **Counts come back as strings.** ``sum_totalTokens`` is ``"25194"``, not
  ``25194``. Adding those together concatenates them.

Langfuse only records the opaque user id, so rows are joined against the
account table: an admin needs to see an address, not a hex string.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.logging import get_logger
from app.core.user_store import list_all_users

logger = get_logger(__name__)

#: Measure names as the metrics API spells them.
_TOKENS = "totalTokens"
_COST = "totalCost"

#: What a row with no userId is called. Document ingestion and evaluation
#: runs produce these, and their spend is real.
UNATTRIBUTED = "Not tied to a user"


class UsageUnavailable(RuntimeError):
    """Langfuse could not answer. Distinct from "there is nothing to report"."""


def _window(days: int) -> tuple[datetime, datetime]:
    """The reporting window, as timezone-aware datetimes.

    Two APIs, two spellings, and sending the wrong one fails only against
    the live service: the trace client reads ``.tzinfo`` off what it is
    given, so an ISO string raises "'str' object has no attribute 'tzinfo'",
    while the metrics endpoint takes JSON and therefore wants the string.
    ``_iso`` converts at the one place that needs it.
    """
    now = datetime.now(timezone.utc)
    return now - timedelta(days=max(1, days)), now


def _iso(window: tuple[datetime, datetime]) -> tuple[str, str]:
    """The same window, spelled the way the metrics API wants it."""
    return window[0].isoformat(), window[1].isoformat()


#: Sent with every call, because the SDK's own default is to retry with
#: backoff and that is the wrong behaviour here. Langfuse allows 100 API
#: requests a day and answers 429 with a ``retry-after`` of the better part
#: of a day, so a retry cannot possibly succeed - it just sat on the
#: connection for 122 seconds, holding a worker thread, before giving up and
#: leaving the page blank anyway. Failing immediately lets the table render
#: with the token column unknown, which is the honest answer.
REQUEST_OPTIONS: dict[str, Any] = {"max_retries": 0, "timeout_in_seconds": 20}


def _run(lf: Any, query: dict) -> list[dict[str, Any]]:
    """Send one metrics query. Raises UsageUnavailable on any failure."""
    try:
        response = lf.api.metrics.metrics(
            query=json.dumps(query), request_options=REQUEST_OPTIONS)
    except Exception as e:  # noqa: BLE001 - the SDK raises its own hierarchy
        logger.exception("langfuse_metrics_failed", view=query.get("view"))
        raise UsageUnavailable(str(e)) from e
    return list(getattr(response, "data", None) or [])


def _as_int(value: Any) -> int:
    """Coerce "25194", 25194, 25194.0 or None to an int."""
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        logger.warning("langfuse_metric_not_a_number", value=repr(value))
        return 0


def _as_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("langfuse_metric_not_a_number", value=repr(value))
        return 0.0


#: The most Langfuse will accept for ``config.row_limit``. Above this the
#: query is refused as "Invalid request data" - measured, not documented:
#: 1000 is served, 5000 is rejected. Because the token query fails soft, a
#: limit set too high shows up only as every user reporting zero tokens
#: beside a correct cost, which is a maddening thing to debug.
MAX_ROW_LIMIT = 1000

#: How many pages of traces one call will walk, at 100 per page. Capped to
#: match MAX_ROW_LIMIT: reading more traces than the token query can cover
#: would leave a tail of rows whose tokens read 0 for no visible reason.
MAX_TRACE_PAGES = 10
_TRACE_PAGE_SIZE = 100


def _traces_in_window(lf: Any, days: int) -> list[Any]:
    """Every trace in the window, up to the page cap."""
    # Datetimes, not strings: the trace client reads .tzinfo off these.
    frm, to = _window(days)
    collected: list[Any] = []
    for page in range(1, MAX_TRACE_PAGES + 1):
        try:
            result = lf.api.trace.list(
                page=page, limit=_TRACE_PAGE_SIZE,
                from_timestamp=frm, to_timestamp=to,
                request_options=REQUEST_OPTIONS,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("langfuse_trace_page_failed", page=page)
            raise UsageUnavailable(str(e)) from e
        data = list(getattr(result, "data", None) or [])
        collected.extend(data)
        total_pages = getattr(getattr(result, "meta", None), "total_pages", 1) or 1
        if page >= total_pages or len(data) < _TRACE_PAGE_SIZE:
            break
    return collected


def _grouped_token_rows(lf: Any, days: int) -> list[dict[str, Any]]:
    """One row per trace in the window, with its token total.

    A single query, and the only one anything here needs to attribute tokens.
    Langfuse allows 100 API requests a day, so a query per trace is not an
    option: one page of twenty rows fetched individually spends a fifth of
    the daily quota, and once it is gone every figure on the page reads
    "unavailable" until the limit resets the next morning.
    """
    frm, to = _iso(_window(days))
    return _run(lf, {
        "view": "observations",
        "dimensions": [{"field": "traceId"}],
        "metrics": [{"measure": _TOKENS, "aggregation": "sum"}],
        # traceId is high cardinality: both of these or Langfuse refuses.
        "orderBy": [{"field": f"sum_{_TOKENS}", "direction": "desc"}],
        "config": {"row_limit": MAX_ROW_LIMIT},
        "fromTimestamp": frm,
        "toTimestamp": to,
    })


def _counts_from(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(r.get("traceId") or ""): _as_int(r.get(f"sum_{_TOKENS}"))
        for r in rows
        if r.get("traceId")
    }


def _tokens_by_trace(lf: Any, days: int) -> dict[str, int]:
    """Tokens for each trace, keyed by trace id.

    Best effort. Cost is the number an admin is actually asking about, and a
    metrics outage must not empty the table.
    """
    try:
        rows = _grouped_token_rows(lf, days)
    except UsageUnavailable:
        logger.warning("token_attribution_unavailable")
        return {}
    return _counts_from(rows)


def tokens_for_traces(
    lf: Any, trace_ids: list[str], days: int = 30,
) -> dict[str, int | None]:
    """Tokens for each of the given traces: a count, or None if unknown.

    None and 0 are different answers and the table draws them differently.
    A trace that spent nothing - a greeting, answered without a model call -
    produces no observations and so is simply absent from the grouping, which
    makes absence mean "free" as long as the grouping is complete. At the row
    limit it was truncated instead, and calling that tail free would
    understate the bill in the exact place an admin looks for overspend.
    """
    ids = [str(i) for i in trace_ids if i]
    if not ids:
        # An empty table must not spend one of the day's hundred requests to
        # find out that it is empty.
        return {}
    try:
        rows = _grouped_token_rows(lf, days)
    except UsageUnavailable:
        logger.warning("page_token_attribution_unavailable", traces=len(ids))
        return {i: None for i in ids}

    counts = _counts_from(rows)
    truncated = len(rows) >= MAX_ROW_LIMIT
    missing: int | None = None if truncated else 0
    return {i: counts.get(i, missing) for i in ids}


def per_user_usage(lf: Any, days: int = 30, limit: int = 50) -> list[dict[str, Any]]:
    """Questions, tokens and spend for each user, dearest first.

    Built from two sources because neither one can answer alone, and the
    obvious single query is quietly wrong:

    * Grouping observations by ``userId`` returns one row with everything in
      it. The SDK does not carry a trace's user down to the generations
      beneath it, so every observation has ``user_id=""`` and the whole bill
      lands under "not tied to a user".
    * Traces carry the user and the cost, but no token count at all.
    * Observations carry the tokens, and the trace they belong to.

    So spend and question counts are summed from the traces, and tokens are
    attributed by joining the per-trace token totals onto them.
    """
    traces = _traces_in_window(lf, days)
    tokens_by_trace = _tokens_by_trace(lf, days)

    totals_by_user: dict[str, dict[str, Any]] = {}
    for t in traces:
        user_id = getattr(t, "user_id", "") or ""
        bucket = totals_by_user.setdefault(
            user_id, {"cost": 0.0, "tokens": 0, "traces": 0})
        bucket["cost"] += _as_float(getattr(t, "total_cost", 0))
        bucket["tokens"] += tokens_by_trace.get(str(getattr(t, "id", "")), 0)
        bucket["traces"] += 1

    names = {u["user_id"]: u for u in list_all_users()}
    out: list[dict[str, Any]] = []
    for user_id, bucket in totals_by_user.items():
        account = names.get(user_id)
        if account:
            username, role = account["username"], account.get("role", "user")
        elif user_id:
            # The traces outlive the account. Hiding the row would quietly
            # remove its spend from what the admin is looking at.
            username, role = f"{user_id[:8]} (deleted)", ""
        else:
            username, role = UNATTRIBUTED, ""
        out.append({
            "user_id": user_id,
            "username": username,
            "role": role,
            "traces": bucket["traces"],
            "tokens": bucket["tokens"],
            "cost": round(bucket["cost"], 8),
        })

    out.sort(key=lambda r: r["cost"], reverse=True)
    return out[:max(1, limit)]


def totals(lf: Any, days: int = 30) -> dict[str, Any]:
    """Tokens and spend across the window, with no grouping."""
    frm, to = _iso(_window(days))
    rows = _run(lf, {
        "view": "observations",
        # No dimension: one row back, or none at all for an empty project.
        "dimensions": [],
        "metrics": [
            {"measure": _TOKENS, "aggregation": "sum"},
            {"measure": _COST, "aggregation": "sum"},
        ],
        "fromTimestamp": frm,
        "toTimestamp": to,
    })
    if not rows:
        return {"tokens": 0, "cost": 0.0}
    return {
        "tokens": _as_int(rows[0].get(f"sum_{_TOKENS}")),
        "cost": _as_float(rows[0].get(f"sum_{_COST}")),
    }


