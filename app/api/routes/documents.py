"""Document upload and management endpoints."""

import asyncio
import re
from pathlib import Path

from fastapi import APIRouter, Cookie, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from app.config import get_settings
from app.core.auth import require_authenticated_user
from app.core.document_loader import (
    SUPPORTED_EXTENSIONS,
    chunk_documents,
    load_document_multimodal,
    save_upload,
)
from app.core.evaluator import generate_golden_for_document
from app.core.logging import get_logger
from app.core.observability import create_trace
from app.core.user_store import delete_document, get_document, get_user_documents, get_user_stats, record_document
from app.core.vector_store import add_documents, delete_documents_by_source, get_collection_stats
from app.models.schemas import (
    CollectionStatsResponse,
    DocumentUploadResponse,
    UserDocumentInfo,
    UserDocumentsResponse,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

# File signatures (magic bytes) for supported binary formats
_MAGIC_SIGNATURES: dict[str, list[bytes]] = {
    ".pdf": [b"%PDF"],
    ".docx": [b"PK\x03\x04"],  # ZIP-based Office format
    ".doc": [b"\xd0\xcf\x11\xe0"],  # OLE2
    ".png": [b"\x89PNG"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".gif": [b"GIF87a", b"GIF89a"],
    ".bmp": [b"BM"],
    ".webp": [b"RIFF"],
    ".tiff": [b"II\x2a\x00", b"MM\x00\x2a"],
    ".tif": [b"II\x2a\x00", b"MM\x00\x2a"],
}


def _validate_magic_bytes(content: bytes, ext: str) -> bool:
    """Verify that file content matches the expected magic bytes for its extension."""
    sigs = _MAGIC_SIGNATURES.get(ext)
    if sigs is None:
        # Text formats (.txt, .md) have no magic bytes — allow
        return True
    return any(content.startswith(sig) for sig in sigs)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(require_authenticated_user),
) -> DocumentUploadResponse:
    """Upload a document, chunk it, and index into the vector store."""
    x_user_id = current_user["user_id"]
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    # Read and validate size
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")

    # Validate magic bytes match the declared extension
    if not _validate_magic_bytes(content, ext):
        raise HTTPException(
            status_code=400,
            detail=f"File content does not match '{ext}' format",
        )

    # Reset file pointer and save under user-specific directory
    await file.seek(0)
    file_path = await save_upload(file.file, file.filename, user_id=x_user_id)

    # ── Langfuse trace for document ingestion ────────────────────
    trace = create_trace(
        name="document-ingest",
        user_id=x_user_id,
        session_id=x_user_id,
        input={"filename": file.filename, "size_bytes": len(content)},
        tags=["ingest", "document"],
        metadata={"extension": ext},
    )

    try:
        preprocess_span = trace.span(name="preprocess", input={"file": str(file_path)})
        # Off the event loop: preprocessing is a long, blocking mix of file I/O
        # and (now concurrent) vision calls. Running it inline would stall every
        # other request for the whole ingest.
        raw_docs, image_records = await asyncio.to_thread(
            load_document_multimodal, str(file_path)
        )
        preprocess_span.update(output={
            "text_docs": len(raw_docs),
            "visual_elements": len(image_records),
        })
        preprocess_span.end()

        chunk_span = trace.span(name="chunking", input={"text_docs": len(raw_docs)})
        parents, children = await asyncio.to_thread(chunk_documents, raw_docs)
        chunk_span.update(output={"parents": len(parents), "children": len(children)})
        chunk_span.end()

        index_span = trace.span(name="indexing", input={
            "children": len(children), "parents": len(parents),
            "images": len(image_records),
        })
        ids = add_documents(
            child_docs=children,
            parent_docs=parents,
            image_records=image_records,
            user_id=x_user_id,
        )
        index_span.update(output={"indexed_ids": len(ids)})
        index_span.end()
    except Exception:
        logger.exception("document_processing_error", filename=file.filename)
        trace.update(output={"error": "processing_failed"})
        trace.end()
        raise HTTPException(status_code=500, detail="Failed to process document")

    trace.update(output={
        "chunks_added": len(ids),
        "images_extracted": len(image_records),
    })
    trace.end()

    # Record in user document history
    if x_user_id:
        record_document(
            user_id=x_user_id,
            filename=file.filename,
            file_size=len(content),
            chunks_added=len(ids),
            images_extracted=len(image_records),
        )

    # Auto-generate golden dataset from this document (background)
    generate_golden_for_document(
        parent_chunks=parents,
        source=file.filename,
        user_id=x_user_id,
    )

    return DocumentUploadResponse(
        filename=file.filename,
        chunks_added=len(ids),
        images_extracted=len(image_records),
        message=(
            f"Indexed {len(parents)} parent + {len(children)} child chunks, "
            f"{len(image_records)} visual elements (multimodal)"
        ),
    )


@router.get("/history", response_model=UserDocumentsResponse)
async def user_document_history(
    current_user: dict = Depends(require_authenticated_user),
) -> UserDocumentsResponse:
    """Return documents previously uploaded by this user."""
    x_user_id = current_user["user_id"]
    docs = get_user_documents(x_user_id)
    stats = get_user_stats(x_user_id)
    return UserDocumentsResponse(
        user_id=x_user_id,
        documents=[UserDocumentInfo(**d) for d in docs],
        stats=stats,
    )


@router.delete("/{doc_id}")
async def delete_user_document(
    doc_id: str,
    current_user: dict = Depends(require_authenticated_user),
):
    """Delete a document and all its chunks from the vector store."""
    x_user_id = current_user["user_id"]
    doc = get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc["user_id"] != x_user_id:
        raise HTTPException(status_code=403, detail="Not authorised")

    removed = delete_documents_by_source(doc["filename"], user_id=x_user_id)
    delete_document(doc_id)

    # Remove uploaded file(s) from disk
    settings = get_settings()
    upload_dir = Path(settings.upload_dir) / x_user_id
    if upload_dir.is_dir():
        for f in upload_dir.iterdir():
            if f.name == doc["filename"] and f.is_file():
                f.unlink()
                logger.info("file_removed", path=str(f))

    # Remove extracted images (e.g. PDF visual elements) with stem prefix
    extracted_dir = Path(settings.upload_dir) / "extracted"
    stem = Path(doc["filename"]).stem
    if extracted_dir.is_dir():
        for f in extracted_dir.iterdir():
            if f.is_file() and f.name.startswith(stem):
                f.unlink()
                logger.info("extracted_file_removed", path=str(f))

    logger.info("document_deleted", doc_id=doc_id, filename=doc["filename"], chunks_removed=removed)
    return {"status": "deleted", "doc_id": doc_id, "chunks_removed": removed}


_FIGURE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def _figure_of(stem: str, name: str) -> bool:
    """True if *name* is a figure preprocessing extracted from a document named *stem*.

    Matches the three names preprocessing writes ({stem}_p3_chart.png,
    {stem}_img2.png, {stem}_smartart1.png) exactly, so a document called
    "report" does not claim the figures of "report_2024".
    """
    return re.match(rf"^{re.escape(stem)}_(p\d+_|img\d+\.|smartart\d+\.)", name) is not None


def _owned_figure(raw_path: str, user_id: str) -> Path | None:
    """The figure at *raw_path* if *user_id* owns it, else None.

    Ownership comes from the database, never from the path: an uploaded image
    must sit in the user's own folder, and an extracted figure must be named
    after one of the user's documents. Anything outside the upload folder,
    missing, or not an image is None, so every refusal looks the same.
    """
    root = Path(get_settings().upload_dir).resolve()
    try:
        path = Path(raw_path).resolve()
    except (OSError, ValueError):
        return None
    if not path.is_relative_to(root) or path.suffix.lower() not in _FIGURE_TYPES or not path.is_file():
        return None
    parts = path.relative_to(root).parts
    if len(parts) != 2:
        return None
    folder, name = parts
    if folder == user_id:
        return path
    if folder == "extracted" and any(_figure_of(Path(d["filename"]).stem, name) for d in get_user_documents(user_id)):
        return path
    return None


@router.get("/figure")
async def document_figure(path: str, current_user: dict = Depends(require_authenticated_user)) -> FileResponse:
    """Serve an image an answer cited (the `images` of the chat stream's meta event)."""
    figure = _owned_figure(path, current_user["user_id"])
    if figure is None:
        raise HTTPException(status_code=404, detail="Figure not found")
    return FileResponse(
        figure,
        media_type=_FIGURE_TYPES[figure.suffix.lower()],
        # Private: it is one user's document. Figures never change once extracted.
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.get("/stats", response_model=CollectionStatsResponse)
async def collection_stats(
    current_user: dict = Depends(require_authenticated_user),
) -> CollectionStatsResponse:
    """Return basic stats about the vector store."""
    stats = get_collection_stats()
    return CollectionStatsResponse(**stats)
