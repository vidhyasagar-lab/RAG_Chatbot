"""Admin API endpoints — user management, Langfuse insights, and RAG evaluation."""

from __future__ import annotations

from fastapi import APIRouter, Cookie, Depends, HTTPException

from app.core.auth import require_admin_user
from app.core.logging import get_logger
from app.core.observability import _get_langfuse
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

@router.get("/langfuse/traces")
async def langfuse_traces(
    admin: dict = Depends(require_admin_user),
    page: int = 1,
    limit: int = 20,
):
    """Fetch recent traces from Langfuse."""
    lf = _get_langfuse()
    if not lf:
        return {"traces": [], "total": 0, "message": "Langfuse not configured"}
    try:
        result = lf.api.trace.list(page=page, limit=limit)
        traces = []
        for t in result.data:
            traces.append({
                "id": t.id,
                "name": t.name,
                "user_id": getattr(t, "user_id", ""),
                "session_id": getattr(t, "session_id", ""),
                "input": str(getattr(t, "input", "") or ""),
                "output": str(getattr(t, "output", "") or ""),
                "tags": getattr(t, "tags", []),
                "created_at": str(getattr(t, "timestamp", "")),
                "total_cost": getattr(t, "total_cost", 0) or 0,
                "latency": getattr(t, "latency", 0) or 0,
                "usage": {
                    "input": getattr(getattr(t, "usage", None), "input", 0) or 0,
                    "output": getattr(getattr(t, "usage", None), "output", 0) or 0,
                    "total": getattr(getattr(t, "usage", None), "total", 0) or 0,
                },
            })
        total = getattr(result.meta, "total_items", len(traces))
        return {"traces": traces, "total": total}
    except Exception as e:
        logger.exception("langfuse_traces_fetch_failed")
        return {"traces": [], "total": 0, "error": str(e)}


@router.get("/langfuse/summary")
async def langfuse_summary(admin: dict = Depends(require_admin_user)):
    """Fetch aggregated cost/usage summary from Langfuse."""
    lf = _get_langfuse()
    if not lf:
        return {"enabled": False}
    try:
        # Get recent traces to compute summary
        result = lf.api.trace.list(page=1, limit=100)
        total_cost = 0.0
        total_tokens = 0
        trace_count = len(result.data)
        for t in result.data:
            total_cost += getattr(t, "total_cost", 0) or 0
            usage = getattr(t, "usage", None)
            if usage:
                total_tokens += getattr(usage, "total", 0) or 0
        return {
            "enabled": True,
            "trace_count": trace_count,
            "total_cost": round(total_cost, 6),
            "total_tokens": total_tokens,
        }
    except Exception as e:
        logger.exception("langfuse_summary_failed")
        return {"enabled": True, "error": str(e)}


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

