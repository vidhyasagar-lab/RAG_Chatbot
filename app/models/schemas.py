"""Pydantic request / response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Chat ──────────────────────────────────────────────────────────────

class ChatMessageSchema(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=50_000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10_000, description="User question")
    chat_history: list[ChatMessageSchema] = Field(
        default_factory=list,
        max_length=20,
        description="Recent conversation turns (max 20)",
    )
    top_k: int | None = Field(None, ge=1, le=20)
    user_id: str = Field("", description="User ID for user-scoped retrieval")
    session_id: str = Field("", description="Chat session ID for persistent history")


class SourceInfo(BaseModel):
    source: str
    page: str | int
    chunk_index: int
    content_type: str = "text"  # text | image | table | flowchart | chart | diagram


class ImageInfo(BaseModel):
    path: str
    page: str | int = ""
    source: str = ""
    content_type: str = "image"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    images: list[ImageInfo] = []
    usage: UsageInfo
    trace_id: str = ""
    session_id: str = ""


# ── Documents ─────────────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    filename: str
    chunks_added: int
    images_extracted: int = 0
    message: str


class CollectionStatsResponse(BaseModel):
    total_documents: int
    collection_name: str
    bm25_indexed: int = 0
    parent_chunks_cached: int = 0
    images_indexed: int = 0


# ── Health ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


# ── Feedback ──────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., min_length=1, description="Langfuse trace ID")
    score: float = Field(..., ge=0, le=1, description="0 = negative, 1 = positive")
    comment: str = Field("", max_length=1000)


# ── Users ─────────────────────────────────────────────────────────────

class UserLoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100, pattern=r"^[\w\-. ]+$")


#: An address is the identifier for anyone signing up through the site, so
#: that every account can be sent a code. Deliberately permissive: the real
#: test of an address is whether its owner can read what we send there.
EMAIL_PATTERN = r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$"


class AuthCredentials(BaseModel):
    """Login/registration body.

    The identifier is an email address. It is still called ``username``
    because that is the column, the cookie subject and the field every
    existing client sends; renaming it would be a breaking change to the API
    for a cosmetic gain.

    The address requirement lives here, at the boundary where people sign
    themselves up, rather than in ``register_user`` — the store is also used
    by the admin CLI, which legitimately creates named service accounts that
    are not addresses.

    Password has a floor but no ceiling-side rules here: the authoritative
    check is register_user's (>= 8 chars), and duplicating it would let the
    two disagree. min_length=1 only rejects an absent password outright so
    the 8-char message comes from one place.
    """

    username: str = Field(..., min_length=3, max_length=254, pattern=EMAIL_PATTERN)
    password: str = Field(..., min_length=1, max_length=1024)


class RegisterCredentials(AuthCredentials):
    """Registration body: credentials plus proof of the address.

    ``code`` is required, and that is the whole point. Registration used to
    create an account from nothing but a body, and it is one of the few paths
    that bypass the API key - so anyone who could reach the host could mint
    accounts, each carrying a lifetime answer budget that costs real money to
    serve. Signing in by code has always verified the address before creating
    an account; this makes the password door do the same.

    Required rather than optional: an optional field that callers may omit is
    not a gate, and the old unverified request would have kept working.
    """

    code: str = Field(..., min_length=4, max_length=12)


class EmailCodeRequest(BaseModel):
    """Ask for a sign-in code to be emailed."""

    email: str = Field(..., min_length=3, max_length=254, pattern=EMAIL_PATTERN)


class EmailCodeVerify(BaseModel):
    """Redeem a code. Six digits, whitespace tolerated on the way in."""

    email: str = Field(..., min_length=3, max_length=254, pattern=EMAIL_PATTERN)
    code: str = Field(..., min_length=4, max_length=12)


class EmailCodeSent(BaseModel):
    """What the request endpoint admits to.

    Never says whether the address has an account: that would turn this into
    a way to find out who has one. ``delivered`` is false only when the
    deployment is in development with no mail configured, where the code is
    in the server log instead.
    """

    sent: bool = True
    delivered: bool = True
    expires_in_minutes: int
    resend_in_seconds: int


class AuthUserResponse(BaseModel):
    user_id: str
    username: str
    role: str
    created_at: str
    # Quota, so the client can show what is left and stop before a 403.
    # A limit of None means unmetered (admins); it is not the same as 0.
    exchanges_used: int = 0
    exchanges_limit: int | None = None
    documents_used: int = 0
    documents_limit: int | None = None


class UserResponse(BaseModel):
    user_id: str
    username: str
    created_at: str


class UserDocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_size: int
    chunks_added: int
    images_extracted: int = 0
    status: str
    uploaded_at: str


class UserDocumentsResponse(BaseModel):
    user_id: str
    documents: list[UserDocumentInfo]
    stats: dict


# ── Chat Sessions ─────────────────────────────────────────────────────

class ChatSessionInfo(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str


class StoredChatMessage(BaseModel):
    """A turn as stored. Assistant turns carry what the answer was built on;
    turns stored before that was recorded have only role and content."""

    role: str
    content: str
    sources: list[dict[str, Any]] | None = None
    images: list[dict[str, Any]] | None = None
    trace_id: str | None = None
    eval: dict[str, Any] | None = None


class ChatSessionMessages(BaseModel):
    session_id: str
    title: str
    messages: list[StoredChatMessage]


class RenameSessionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
