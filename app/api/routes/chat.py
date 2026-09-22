"""Chat endpoints – the core RAG conversation API."""

from fastapi import APIRouter, Cookie, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse

from app.core.auth import get_current_user_id, require_authenticated_user
from app.core.chat_store import (
    add_message,
    create_session,
    delete_session,
    get_recent_messages,
    get_session,
    get_session_messages,
    get_user_sessions,
    update_session_title,
)
from app.core.eval_store import get_query_scores
from app.core.logging import get_logger
from app.core.rag_engine import ChatMessage, ask, ask_stream, ask_with_eval
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ChatSessionInfo,
    ChatSessionMessages,
    ImageInfo,
    RenameSessionRequest,
    SourceInfo,
    UsageInfo,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def _auto_title(question: str) -> str:
    """Derive a short session title from the first question."""
    title = question.strip().replace("\n", " ")
    return title[:80] + ("…" if len(title) > 80 else "")


def _ensure_session(request: ChatRequest, user_id: str) -> str:
    """Return the caller's existing session_id, or create a new session.

    A client-supplied session_id is verified against *user_id* before it is
    used. Without that check this function returned whatever the body asked
    for, and the caller then fed that session's history to the model and
    appended to it — so an authenticated user could read another user's
    conversation out of the answer and write into their history.

    *user_id* is a parameter rather than being read from ``request.user_id``
    so the check cannot be defeated by calling this before the route has
    overwritten the body's user_id with the authenticated one.

    404 rather than 403: a wrong owner and a nonexistent session are reported
    identically, so this does not confirm that someone else's session exists.
    """
    if request.session_id:
        session = get_session(request.session_id)
        if not session or session["user_id"] != user_id:
            logger.warning(
                "session_access_denied",
                session_id=request.session_id,
                user_id=user_id,
                exists=bool(session),
            )
            raise HTTPException(status_code=404, detail="Session not found")
        return request.session_id
    session = create_session(user_id, _auto_title(request.question))
    return session["session_id"]


def _load_history(session_id: str) -> list[ChatMessage]:
    """Load recent messages from the DB for LLM context."""
    rows = get_recent_messages(session_id, limit=20)
    return [ChatMessage(role=r["role"], content=r["content"]) for r in rows]


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, current_user: dict = Depends(require_authenticated_user)) -> ChatResponse:
    """Send a question and receive a RAG-augmented answer."""
    # Enforce authenticated user_id instead of trusting request body
    request.user_id = current_user["user_id"]
    session_id = _ensure_session(request, current_user["user_id"])
    is_new = not request.session_id

    # Save user message
    add_message(session_id, "user", request.question)

    # Load history from DB
    history = _load_history(session_id)

    try:
        result = ask(
            question=request.question,
            chat_history=history[:-1],  # exclude current question already in list
            top_k=request.top_k,
            user_id=request.user_id,
        )
    except Exception:
        logger.exception("rag_engine_error")
        raise HTTPException(status_code=502, detail="LLM service error")

    # Save assistant response
    add_message(session_id, "assistant", result.answer)

    # Update title from first question if new session
    if is_new:
        update_session_title(session_id, _auto_title(request.question))

    return ChatResponse(
        answer=result.answer,
        sources=[SourceInfo(**s) for s in result.sources],
        images=[ImageInfo(**img) for img in result.images],
        usage=UsageInfo(**result.usage),
        trace_id=result.trace_id,
        session_id=session_id,
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest, current_user: dict = Depends(require_authenticated_user)) -> StreamingResponse:
    """Stream a RAG-augmented answer via Server-Sent Events (eval-gated)."""
    # Enforce authenticated user_id instead of trusting request body
    request.user_id = current_user["user_id"]
    session_id = _ensure_session(request, current_user["user_id"])
    is_new = not request.session_id

    # Save user message
    add_message(session_id, "user", request.question)

    # Load history from DB
    history = _load_history(session_id)

    async def event_generator():
        try:
            full_answer = ""
            async for chunk in ask_with_eval(
                question=request.question,
                chat_history=history[:-1],
                top_k=request.top_k,
                user_id=request.user_id,
                session_id=session_id,
            ):
                yield chunk
                # Capture full answer from token events
                if chunk.startswith("data: ") and '"type": "token"' in chunk:
                    import json as _json
                    try:
                        payload = _json.loads(chunk[6:])
                        full_answer += payload.get("content", "")
                    except Exception:
                        pass
                elif chunk.startswith("data: ") and '"type": "done"' in chunk:
                    # Save assistant message after streaming completes
                    if full_answer:
                        add_message(session_id, "assistant", full_answer)
                    if is_new:
                        update_session_title(session_id, _auto_title(request.question))
        except Exception:
            logger.exception("rag_stream_error")
            yield 'data: {"type":"error","message":"LLM service error"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Session management endpoints ─────────────────────────────────────

@router.get("/sessions")
async def list_sessions(current_user: dict = Depends(require_authenticated_user)) -> list[ChatSessionInfo]:
    """List chat sessions for the authenticated user."""
    sessions = get_user_sessions(current_user["user_id"])
    return [ChatSessionInfo(**s) for s in sessions]


@router.get("/sessions/{session_id}")
async def get_session_detail(session_id: str, current_user: dict = Depends(require_authenticated_user)) -> ChatSessionMessages:
    """Get all messages for a chat session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Not authorised")
    messages = get_session_messages(session_id)
    return ChatSessionMessages(
        session_id=session_id,
        title=session["title"],
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
    )


@router.patch("/sessions/{session_id}")
async def rename_session(session_id: str, body: RenameSessionRequest, current_user: dict = Depends(require_authenticated_user)):
    """Rename a chat session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Not authorised")
    update_session_title(session_id, body.title)
    return {"ok": True}


@router.delete("/sessions/{session_id}")
async def remove_session(session_id: str, current_user: dict = Depends(require_authenticated_user)):
    """Delete a chat session and all its messages."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Not authorised")
    delete_session(session_id)
    return {"ok": True}


@router.get("/scores/{trace_id}")
async def get_scores(
    trace_id: str,
    current_user: dict = Depends(require_authenticated_user),
):
    """Poll RAGAS quality scores for a given trace. Returns 204 if not yet available."""
    # Someone else's trace reads as 204, indistinguishable from "not ready".
    scores = get_query_scores(trace_id, user_id=current_user["user_id"])
    if scores is None:
        return Response(status_code=204)
    return scores
