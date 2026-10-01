"""Admin API endpoints — user management, Langfuse insights, and RAG evaluation."""

from __future__ import annotations

from fastapi import APIRouter, Cookie, Depends, HTTPException

from app.core.auth import require_admin_user
from app.core.logging import get_logger
from app.core.observability import _get_langfuse
from app.core.usage_stats import (
    REQUEST_OPTIONS,
    UsageUnavailable,
    per_user_usage,
    tokens_for_traces,
    totals,
)
from app.core.user_store import (
    admin_create_user,
    admin_delete_user,
    get_system_stats,
    get_user,
    list_all_users,
    set_user_role,
)
from app.core.vector_store import get_collection_stats
from app.core.eval_store import (
    add_golden_sample,
    clear_golden_dataset,
    delete_golden_sample,
    get_eval_results,
    get_eval_run,
    get_eval_runs,
    get_golden_dataset,
)
from app.core.evaluator import generate_synthetic_dataset, start_evaluation

logger = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ── Users ────────────────────────────────────────────────────────────

@router.get("/users")
async def get_all_users(admin: dict = Depends(require_admin_user)):
    return list_all_users()


@router.post("/users")
async def create_user(
    body: dict,
    admin: dict = Depends(require_admin_user),
):
    username = body.get("username", "").strip()
    password = body.get("password", "")
    role = body.get("role", "user")
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")
    if role not in ("user", "admin"):
        raise HTTPException(status_code=400, detail="Role must be 'user' or 'admin'")
    try:
        user = admin_create_user(username, password, role)
        logger.info("admin_user_created", actor=admin["username"], target=username, role=role)
        return {"status": "created", "user": user}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/users/{target_user_id}")
async def delete_user(target_user_id: str, admin: dict = Depends(require_admin_user)):
    if target_user_id == admin["user_id"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    target = get_user(target_user_id)
    if not admin_delete_user(target_user_id):
        raise HTTPException(status_code=404, detail="User not found")
    logger.info("admin_user_deleted", actor=admin["username"], target_id=target_user_id, target_name=target["username"] if target else "unknown")
    return {"status": "deleted"}


@router.patch("/users/{target_user_id}/role")
async def update_role(target_user_id: str, body: dict, admin: dict = Depends(require_admin_user)):
    if target_user_id == admin["user_id"]:
        raise HTTPException(status_code=400, detail="Cannot change your own role")
    role = body.get("role", "")
    if role not in ("user", "admin"):
        raise HTTPException(status_code=400, detail="Role must be 'user' or 'admin'")
    if not set_user_role(target_user_id, role):
        raise HTTPException(status_code=404, detail="User not found")
    logger.info("admin_role_changed", actor=admin["username"], target_id=target_user_id, new_role=role)
    return {"status": "updated"}


# ── Dashboard stats ──────────────────────────────────────────────────

@router.get("/stats")
async def dashboard_stats(admin: dict = Depends(require_admin_user)):
    sys_stats = get_system_stats()
    vec_stats = get_collection_stats()
    return {**sys_stats, **vec_stats}


# ── Langfuse insights ───────────────────────────────────────────────

#: The only orderings Langfuse will sort a trace list by. Measured against
#: the live API: totalCost, latency and anything else come back "Invalid
#: order by column", so cost and latency cannot be sorted server-side at all.
#: Sorting only the page on screen would be a claim about the other pages
#: that it cannot support, so the UI does not offer it either.
SORTABLE_TRACE_ORDERS = frozenset({
    "timestamp.desc", "timestamp.asc", "name.asc", "name.desc",
})

@router.get("/langfuse/traces")
def langfuse_traces(
    admin: dict = Depends(require_admin_user),
    page: int = 1,
    limit: int = 20,
    user_id: str = "",
    name: str = "",
    order_by: str = "",
    token_days: int = 90,
):
    """Recent traces from Langfuse, filtered and sorted by the service.

    Declared `def`, not `async def`, and so are the three below it. The
    Langfuse SDK is synchronous: called from a coroutine it blocks uvicorn's
    single event loop for the whole round trip, which stalls every other
    request - sign-in included - while the usage page loads. A plain `def`
    route runs in FastAPI's threadpool instead.

    The filters go to Langfuse rather than being applied to the page we
    already fetched: the list is paginated server-side, so filtering here
    would only ever filter the twenty rows on screen.

    Token counts come with the page, in one grouped metrics query, because a
    trace object in SDK 4.x carries no usage at all - tokens live on the
    observations beneath it. They were once fetched per row as it was
    expanded, which was correct but unaffordable: Langfuse allows 100 API
    requests a day, so clicking through two pages of rows spent half the
    quota and left every figure on the page reading "unavailable" until the
    limit reset the next morning.

    ``token_days`` is the window that query covers, and is deliberately
    wider than the summary's: this table can be paged back further than 30
    days, and a trace whose observations fell outside the window would
    report 0 tokens rather than its real count.
    """
    lf = _get_langfuse()
    if not lf:
        return {"traces": [], "total": 0, "message": "Langfuse not configured"}
    try:
        # Only send the filters that were asked for. user_id="" means "a user
        # whose id is the empty string" to Langfuse, which matches nothing.
        # No retries: a 429 here carries a retry-after of nearly a day, so
        # the SDK's backoff only holds the connection and the worker thread.
        filters: dict = {"page": page, "limit": limit,
                         "request_options": REQUEST_OPTIONS}
        if user_id:
            filters["user_id"] = user_id
        if name:
            filters["name"] = name
        if order_by in SORTABLE_TRACE_ORDERS:
            filters["order_by"] = order_by
        elif order_by:
            # Dropped, not forwarded. Langfuse answers "Invalid order by
            # column" for anything else, and this route reports a failure as
            # an empty list - so one unsupported sort blanked the table.
            logger.info("langfuse_order_by_ignored", order_by=order_by)

        result = lf.api.trace.list(**filters)
        traces = []
        for t in result.data:
            traces.append({
                "id": t.id,
                "name": t.name,
                "user_id": getattr(t, "user_id", "") or "",
                "session_id": getattr(t, "session_id", "") or "",
                "input": str(getattr(t, "input", "") or ""),
                "output": str(getattr(t, "output", "") or ""),
                "tags": getattr(t, "tags", []) or [],
                "created_at": str(getattr(t, "timestamp", "")),
                "total_cost": getattr(t, "total_cost", 0) or 0,
                "latency": getattr(t, "latency", 0) or 0,
            })

        # One query for the page. Fails soft to None per row: losing the
        # token column must not cost the table.
        tokens = tokens_for_traces(lf, [t["id"] for t in traces], days=token_days)
        for row in traces:
            row["tokens"] = tokens.get(row["id"])

        total = getattr(result.meta, "total_items", len(traces))
        return {"traces": traces, "total": total}
    except Exception as e:
        logger.exception("langfuse_traces_fetch_failed")
        return {"traces": [], "total": 0, "error": str(e)}


@router.get("/langfuse/by-user")
def langfuse_by_user(
    admin: dict = Depends(require_admin_user),
    days: int = 30,
    limit: int = 50,
):
    """Tokens and spend for each user, dearest first.

    One metrics query however many users there are, with the ids resolved to
    the addresses they belong to - Langfuse stores only an opaque id, which
    an admin cannot act on.
    """
    lf = _get_langfuse()
    if not lf:
        return {"enabled": False, "users": []}
    try:
        return {"enabled": True, "users": per_user_usage(lf, days=days, limit=limit)}
    except UsageUnavailable as e:
        # Reported rather than returned as an empty list: "nobody has spent
        # anything" and "we could not ask" are different answers.
        return {"enabled": True, "users": [], "error": str(e)}


@router.get("/langfuse/summary")
def langfuse_summary(admin: dict = Depends(require_admin_user), days: int = 30):
    """Spend and token totals across the window."""
    lf = _get_langfuse()
    if not lf:
        return {"enabled": False}
    try:
        agg = totals(lf, days=days)
    except UsageUnavailable as e:
        return {"enabled": True, "error": str(e)}

    # The trace count is a separate question: metrics counts observations,
    # and an admin asking "how many questions" means traces.
    trace_count = 0
    try:
        meta = lf.api.trace.list(page=1, limit=1, request_options=REQUEST_OPTIONS).meta
        trace_count = getattr(meta, "total_items", 0)
    except Exception:
        logger.exception("langfuse_trace_count_failed")

    return {
        "enabled": True,
        "trace_count": trace_count,
        "total_cost": round(agg["cost"], 6),
        "total_tokens": agg["tokens"],
        "days": days,
    }


def _truncate(text: str, max_len: int = 150) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


# ── Golden Dataset ───────────────────────────────────────────────────

@router.get("/golden")
async def list_golden(admin: dict = Depends(require_admin_user)):
    """List all golden dataset samples."""
    return get_golden_dataset()


@router.post("/golden")
async def add_golden(body: dict, admin: dict = Depends(require_admin_user)):
    """Add a single golden sample (manual / from real user query)."""
    question = body.get("question", "").strip()
    ground_truth = body.get("ground_truth", "").strip()
    if not question or not ground_truth:
        raise HTTPException(status_code=400, detail="question and ground_truth required")
    sample = add_golden_sample(
        question=question,
        ground_truth=ground_truth,
        source_doc=body.get("source_doc", ""),
        source=body.get("source", "manual"),
    )
    return {"status": "created", "sample": sample}


@router.delete("/golden/{sample_id}")
async def remove_golden(sample_id: str, admin: dict = Depends(require_admin_user)):
    if not delete_golden_sample(sample_id):
        raise HTTPException(status_code=404, detail="Sample not found")
    return {"status": "deleted"}


@router.delete("/golden")
async def clear_all_golden(admin: dict = Depends(require_admin_user)):
    count = clear_golden_dataset()
    logger.info("admin_golden_cleared", actor=admin["username"], deleted=count)
    return {"status": "cleared", "deleted": count}


@router.post("/golden/generate")
async def generate_golden(body: dict = None, admin: dict = Depends(require_admin_user)):
    """Auto-generate golden dataset from user's documents."""
    count_per_doc = (body or {}).get("count_per_doc", 3)
    count = generate_synthetic_dataset(user_id=admin["user_id"], count_per_doc=count_per_doc)
    return {"status": "generated", "count": count}


# ── Evaluation Runs ──────────────────────────────────────────────────

@router.post("/evaluate")
async def run_evaluation(admin: dict = Depends(require_admin_user)):
    """Start a batch evaluation run (background)."""
    golden = get_golden_dataset()
    if not golden:
        raise HTTPException(status_code=400, detail="No golden dataset. Generate or add samples first.")
    run_id = start_evaluation(user_id=admin["user_id"])
    return {"status": "started", "run_id": run_id, "total_samples": len(golden)}


@router.get("/evaluate/runs")
async def list_eval_runs(admin: dict = Depends(require_admin_user)):
    return get_eval_runs()


@router.get("/evaluate/runs/{run_id}")
async def get_run_detail(run_id: str, admin: dict = Depends(require_admin_user)):
    run = get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    results = get_eval_results(run_id)
    return {"run": run, "results": results}

