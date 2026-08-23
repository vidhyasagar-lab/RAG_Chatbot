"""Pydantic request / response schemas."""

from __future__ import annotations

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


class ChatSessionMessages(BaseModel):
    session_id: str
    title: str
    messages: list[ChatMessageSchema]


class RenameSessionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
