"""Spend, tokens and traces for the admin usage page.

Everything comes from one endpoint, ``/api/public/v2/observations``, because
the two this used to read are the two Langfuse rate-limits hardest, and it
told us so itself in a 429: *"Rate limit exceeded for GET /api/public/traces.
Use GET /api/public/v2/observations?fromStartTime=<from>&toStartTime=<to>
for high-volume reads."*

Measured against the live service:

* ``/api/public/traces`` allows **5 requests a minute**. The page made three
  of them on every load - one for the table, one for the per-user panel, one
  for the summary's count - so it could not be opened twice in a minute, and
  the per-user walk could fire ten in a single request.
* the metrics API allows **100 requests a day**, which capped the page at
  roughly thirty loads before every figure read "unavailable" until the
  following morning.
* the observations endpoint serves 1000 rows a call, pages by cursor, and
  carries the lot.

Two facts about its rows shape everything below, and neither is documented:

* ``user_id`` is set **only on the root span**. The generations beneath it
  have none, which is why grouping observations by ``userId`` in the metrics
  API returned a single bucket with the whole bill in it.
* ``total_cost`` and ``usage_details`` are set **only on the generations**.
  The root span carries no money at all.

So neither row type can answer on its own, and the endpoint's ``user_id``
filter is no shortcut either: it matches per observation, so asking for one
user returns their roots without the children, reporting no spend. Two
reads, joined locally on ``trace_id``, is the only shape that works.

One more trap: ``fields`` defaults to ``core,basic``, and a read without the
``usage`` and ``metrics`` groups comes back with ``total_cost=None`` on every
row - correct-looking, and entirely empty.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.logging import get_logger
from app.core.user_store import list_all_users

logger = get_logger(__name__)

#: What a row with no user is called. Document ingestion and evaluation runs
#: produce these, and the vision preprocessing they do is real money.
UNATTRIBUTED = "Not tied to a user"

#: Field groups to ask for. Without ``usage`` and ``metrics`` the cost and
#: token figures are null on every row; the default is ``core,basic``.
OBSERVATION_FIELDS = "core,basic,io,usage,metrics,time,trace_context"

#: Rows per call. 1000 is the documented maximum, and asking for all of it
#: is what keeps a page load down to two requests.
OBSERVATION_PAGE_SIZE = 1000

#: How many pages of each row type one snapshot will walk. Three is enough
#: for 3,000 observations of each kind; beyond that the snapshot says it
#: truncated rather than quietly reporting part of the bill as the whole.
MAX_OBSERVATION_PAGES = 3

#: Sent with every call. The SDK retries with backoff by default, which is
#: wrong for both limits here: the trace endpoint answers 429 with a
#: retry-after of a minute and the metrics API with nearly a day, so the
#: retry only held a worker thread - 122 seconds, measured - before failing
#: anyway and leaving the page blank.
REQUEST_OPTIONS: dict[str, Any] = {"max_retries": 0, "timeout_in_seconds": 30}


class UsageUnavailable(RuntimeError):
    """Langfuse could not answer. Distinct from "there is nothing to report"."""


def _window(days: int) -> tuple[datetime, datetime]:
    """The reporting window, as timezone-aware datetimes.

    Datetimes, not ISO strings: the client reads ``.tzinfo`` off what it is
    given, and a string fails with "'str' object has no attribute 'tzinfo'".
    """
    now = datetime.now(timezone.utc)
    return now - timedelta(days=max(1, days)), now


def _as_int(value: Any) -> int:
    """Coerce "2420", 2420, 2420.0 or None to an int."""
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        logger.warning("usage_value_not_a_number", value=repr(value))
        return 0


def _as_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("usage_value_not_a_number", value=repr(value))
        return 0.0


def _walk(lf: Any, days: int, **filters: Any) -> tuple[list[Any], bool]:
    """Read observations matching ``filters``, following the cursor.

    Returns the rows and whether the walk stopped at the page cap with more
    left behind. Raises UsageUnavailable rather than returning an empty list:
    "nobody has spent anything" and "we were rate-limited" are different
    answers, and drawing the second as the first is how the page spent a day
    claiming every user had cost nothing.
    """
    frm, to = _window(days)
    rows: list[Any] = []
    cursor: str | None = None
    for _ in range(MAX_OBSERVATION_PAGES):
        try:
            result = lf.api.observations.get_many(
                limit=OBSERVATION_PAGE_SIZE,
                cursor=cursor,
                fields=OBSERVATION_FIELDS,
                from_start_time=frm,
                to_start_time=to,
                request_options=REQUEST_OPTIONS,
                **filters,
            )
        except Exception as e:  # noqa: BLE001 - the SDK raises its own hierarchy
            logger.exception("observations_read_failed", filters=sorted(filters))
            raise UsageUnavailable(str(e)) from e

        rows.extend(list(getattr(result, "data", None) or []))
        cursor = getattr(getattr(result, "meta", None), "cursor", None)
        if not cursor:
            # No cursor means no more rows: the walk finished.
            return rows, False

    logger.warning("observations_walk_truncated", filters=sorted(filters), rows=len(rows))
    return rows, True


def _tokens_of(observation: Any) -> int:
    """The total token count on a generation.

    ``usage_details`` is ``{}`` on spans and can be absent altogether on a
    generation that failed, so neither shape may raise.
    """
    usage = getattr(observation, "usage_details", None) or {}
    if not isinstance(usage, dict):
        return 0
    return _as_int(usage.get("total"))


def usage_snapshot(lf: Any, days: int = 30, user_limit: int = 200) -> dict[str, Any]:
    """Everything the usage page shows, from two reads of one endpoint.

    The whole window is returned rather than a page of it, which is what
    makes the table honest: sorting and paging happen in the browser over
    rows it actually holds, instead of claiming an order over pages it has
    never seen. ``truncated`` says when that stopped being true.
    """
    roots, roots_cut = _walk(lf, days, is_root_observation=True)
    generations, gens_cut = _walk(lf, days, type="GENERATION")

    # Money first, keyed by the trace it belongs to - the only field the two
    # row types share.
    cost_by_trace: dict[str, float] = {}
    tokens_by_trace: dict[str, int] = {}
    for gen in generations:
        trace_id = str(getattr(gen, "trace_id", "") or "")
        if not trace_id:
            continue
        cost_by_trace[trace_id] = cost_by_trace.get(trace_id, 0.0) + _as_float(
            getattr(gen, "total_cost", None))
        tokens_by_trace[trace_id] = tokens_by_trace.get(trace_id, 0) + _tokens_of(gen)

    accounts = {u["user_id"]: u for u in list_all_users()}

    def name_of(user_id: str) -> tuple[str, str]:
        account = accounts.get(user_id)
        if account:
            return account["username"], account.get("role", "user")
        if user_id:
            # The traces outlive the account. Hiding the row would quietly
            # remove its spend from what the admin is looking at.
            return f"{user_id[:8]} (deleted)", ""
        return UNATTRIBUTED, ""

    traces: list[dict[str, Any]] = []
    for r in roots:
        trace_id = str(getattr(r, "trace_id", "") or "")
        if not trace_id:
            continue
        user_id = str(getattr(r, "user_id", "") or "")
        username, _role = name_of(user_id)
        traces.append({
            "id": trace_id,
            "name": str(getattr(r, "trace_name", "") or getattr(r, "name", "") or ""),
            "user_id": user_id,
            "username": username,
            "session_id": str(getattr(r, "session_id", "") or ""),
            "input": str(getattr(r, "input", "") or ""),
            "output": str(getattr(r, "output", "") or ""),
            "tags": list(getattr(r, "tags", None) or []),
            "created_at": str(getattr(r, "start_time", "") or ""),
            "total_cost": round(cost_by_trace.get(trace_id, 0.0), 8),
            "latency": _as_float(getattr(r, "latency", None)),
            # 0, not None: a trace with no generation never reached a model,
            # which small talk does not, and that really was free.
            "tokens": tokens_by_trace.get(trace_id, 0),
        })

    # The join is by dictionary, so the order has to be restored explicitly.
    traces.sort(key=lambda t: t["created_at"], reverse=True)

    per_user: dict[str, dict[str, Any]] = {}
    for t in traces:
        bucket = per_user.setdefault(
            t["user_id"], {"cost": 0.0, "tokens": 0, "traces": 0})
        bucket["cost"] += t["total_cost"]
        bucket["tokens"] += t["tokens"]
        bucket["traces"] += 1

    users = []
    for user_id, bucket in per_user.items():
        username, role = name_of(user_id)
        users.append({
            "user_id": user_id,
            "username": username,
            "role": role,
            "traces": bucket["traces"],
            "tokens": bucket["tokens"],
            "cost": round(bucket["cost"], 8),
        })
    users.sort(key=lambda u: u["cost"], reverse=True)

    # Totals span every generation read, including any whose root fell
    # outside the walk - that spend is real and belongs in the bill even
    # when the question behind it is not on the list.
    return {
        "traces": traces,
        "users": users[:max(1, user_limit)],
        "totals": {
            "cost": round(sum(cost_by_trace.values()), 8),
            "tokens": sum(tokens_by_trace.values()),
            "traces": len(traces),
        },
        "truncated": roots_cut or gens_cut,
        "days": days,
    }
