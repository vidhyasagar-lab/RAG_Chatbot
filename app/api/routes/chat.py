"""Chat endpoints – the core RAG conversation API."""

import json

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
from app.core.quota import check_exchange_quota, is_exempt
from app.core.user_store import increment_exchanges
from app.core.rag_engine import ChatMessage, ask, ask_stream
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
    # Before the session is touched and before anything reaches Azure, so an
    # exhausted budget costs nothing and leaves no half-started conversation.
    check_exchange_quota(current_user)

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
    # Charged on delivery, matching /chat/stream: the 502 above returns before
    # this line, so a failed call does not spend the budget.
    if not is_exempt(current_user):
        increment_exchanges(current_user["user_id"])

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
    """Stream a RAG-augmented answer via Server-Sent Events.

    Tokens are sent as they are generated, whether or not the quality gate is
    on. With ``EVAL_GATING_ENABLED`` the gate runs after the answer has
    streamed and may reject it, in which case a replacement follows:

        meta -> stage -> token+ -> [eval -> [replace -> stage -> token+]] -> done

    Each ``token`` carries the ``attempt`` it belongs to, and ``done`` carries
    ``final_attempt`` - the answer that stands. Only that one is stored.
    """
    # Before the session is touched and before anything reaches Azure, so an
    # exhausted budget costs nothing and leaves no half-started conversation.
    check_exchange_quota(current_user)

    # Enforce authenticated user_id instead of trusting request body
    request.user_id = current_user["user_id"]
    session_id = _ensure_session(request, current_user["user_id"])
    is_new = not request.session_id

    # Save user message
    add_message(session_id, "user", request.question)

    # Load history from DB
    history = _load_history(session_id)

    pipeline_args = dict(
        question=request.question,
        chat_history=history[:-1],
        top_k=request.top_k,
        user_id=request.user_id,
        session_id=session_id,
    )

    # The gated pipeline can stream two answers: a draft the gate rejects, then
    # its replacement. Accumulating them into one string would save the rejected
    # draft glued to the answer that replaced it, so each attempt is kept apart
    # and `done` names the winner.
    answers: dict[int, str] = {}
    # What each answer was built on, stored with it so a reopened chat can show
    # its sources and verdict. One verdict per attempt: a rejected draft's
    # scores must not be filed against the replacement.
    context: dict = {}
    verdicts: dict[int, dict] = {}
    saved = False

    def _persist(attempt: int | None = None) -> None:
        """Write the assistant's answer to the session exactly once.

        `done` is emitted after the quality gate, which takes 25-120s, so
        anything that ends the connection in that window - Stop, navigation, a
        closed tab, a sleeping laptop - would otherwise discard an answer the
        reader has already read in full and that this function is holding.

        Without a `final_attempt` to go on, the newest attempt that produced
        text is the one the reader was looking at when the connection died.
        """
        nonlocal saved
        if saved:
            return
        chosen = attempt if attempt is not None and answers.get(attempt) else None
        if chosen is None:
            chosen = next((a for a in sorted(answers, reverse=True) if answers[a]), None)
        if chosen is None:
            return
        saved = True
        meta = dict(context)
        if chosen in verdicts:
            meta["eval"] = verdicts[chosen]
        add_message(session_id, "assistant", answers[chosen], meta=meta or None)
        # Charged here rather than at the start, so the budget tracks answers
        # the reader actually received: a request that dies before producing
        # text costs nothing, and one they read in full is charged even if
        # they disconnect during the quality gate.
        if not is_exempt(current_user):
            increment_exchanges(current_user["user_id"])
        if is_new:
            update_session_title(session_id, _auto_title(request.question))

    async def event_generator():
        try:
            async for chunk in ask_stream(**pipeline_args):
                yield chunk
                if not chunk.startswith("data: "):
                    continue
                try:
                    payload = json.loads(chunk[6:])
                except ValueError:
                    continue

                kind = payload.get("type")
                if kind == "token":
                    attempt = payload.get("attempt", 1)
                    answers[attempt] = answers.get(attempt, "") + payload.get("content", "")
                elif kind == "meta":
                    context.update(
                        sources=payload.get("sources") or [],
                        images=payload.get("images") or [],
                        trace_id=payload.get("trace_id") or "",
                    )
                elif kind == "eval":
                    attempt = payload.get("attempt", 1)
                    verdicts[attempt] = {
                        **(payload.get("scores") or {}),
                        "verdict": payload.get("verdict"),
                        "attempt": attempt,
                    }
                elif kind == "done":
                    _persist(payload.get("final_attempt", 1))
        except Exception:
            logger.exception("rag_stream_error")
            yield 'data: {"type":"error","message":"LLM service error"}\n\n'
        finally:
            # Covers the disconnect and the mid-pipeline exception alike. A
            # no-op when `done` already persisted.
            _persist()

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


@router.get("/sessions/{session_id}", response_model_exclude_none=True)
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
        messages=[{"role": m["role"], "content": m["content"], **(m.get("meta") or {})} for m in messages],
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
