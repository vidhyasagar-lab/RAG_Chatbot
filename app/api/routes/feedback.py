"""Feedback endpoint — lets users score RAG responses via Langfuse."""

from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import require_authenticated_user
from app.core.eval_store import trace_belongs_to
from app.core.logging import get_logger
from app.core.observability import score_trace
from app.models.schemas import FeedbackRequest

logger = get_logger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("/")
async def submit_feedback(
    request: FeedbackRequest,
    current_user: dict = Depends(require_authenticated_user),
) -> dict:
    """Record user feedback (thumbs up/down) for a chat response."""
    if not trace_belongs_to(request.trace_id, current_user["user_id"]):
        logger.warning(
            "feedback_trace_not_owned",
            trace_id=request.trace_id,
            user_id=current_user["user_id"],
        )
        raise HTTPException(status_code=404, detail="Trace not found")

    score_trace(
        trace_id=request.trace_id,
        name="user-feedback",
        value=request.score,
        comment=request.comment,
    )
    logger.info(
        "feedback_recorded",
        trace_id=request.trace_id,
        score=request.score,
    )
    return {"status": "ok"}
