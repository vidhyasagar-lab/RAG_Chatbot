"""Document loading and advanced hybrid chunking — multimodal.

Supports text documents (PDF, DOCX, TXT, MD) AND images (PNG, JPG, etc.).
Images and PDF-embedded visuals are described via GPT-5.2 vision and
the resulting text is chunked alongside regular document text.

Best-practice chunking pipeline for production multimodal RAG:

  1. **Text cleaning** — normalise whitespace, strip noise.
  2. **Visual extraction** — extract images/tables from PDFs via PyMuPDF,
     describe standalone images via GPT-5.2 vision.
  3. **Structural splitting** — detect Markdown / document headers and
     split by logical sections so headings are never ripped apart.
  4. **Semantic splitting** — within each section, use embedding
     cosine-distance to find natural topic boundaries.
  5. **Token-aware size enforcement** — cap every chunk to a token budget
     (using ``tiktoken``) so nothing blows up the embedding or LLM context.
  6. **Contextual header injection** — prepend section / document title
     to every chunk so it can stand alone without its neighbours.
  7. **Parent / Child hierarchy** — large parent chunks give the LLM
     broad context; small child chunks are embedded for precision.
  8. **Near-duplicate removal** — drop chunks whose content overlaps > 90 %
     with an already-seen chunk.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import BinaryIO

import tiktoken
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from app.config import get_settings
from app.core.embeddings import get_embeddings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Supported MIME → loader mapping
_EXTENSION_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".txt": "text",
    ".md": "text",
    ".docx": "docx",
    ".doc": "docx",
}

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

SUPPORTED_EXTENSIONS = set(_EXTENSION_MAP.keys()) | _IMAGE_EXTENSIONS


def _load_pdf(file_path: str) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(file_path).load()


def _load_text(file_path: str) -> list[Document]:
    from langchain_community.document_loaders import TextLoader

    return TextLoader(file_path, encoding="utf-8").load()


def _load_docx(file_path: str) -> list[Document]:
    from langchain_community.document_loaders import Docx2txtLoader

    return Docx2txtLoader(file_path).load()


_LOADERS = {
    "pdf": _load_pdf,
    "text": _load_text,
    "docx": _load_docx,
}


def _safe_filename(filename: str) -> str:
    """Sanitise a user-supplied filename to prevent path traversal.

    Uses PurePosixPath + PureWindowsPath to handle both separator styles
    regardless of the host OS, then strips any remaining unsafe characters.
    """
    from pathlib import PurePosixPath, PureWindowsPath
    import re as _re

    # Extract just the final component for both Unix and Windows separators
    name = PurePosixPath(PureWindowsPath(filename).name).name
    # Remove any remaining traversal fragments or null bytes
    name = name.replace("\0", "").replace("..", "")
    # Allow only alphanumerics, hyphens, underscores, dots, spaces
    name = _re.sub(r"[^\w.\- ]", "_", name)
    if not name or name.startswith("."):
        name = "upload_" + name
    return name


async def save_upload(file: BinaryIO, filename: str, user_id: str = "") -> Path:
    """Persist an uploaded file to the upload directory and return its path.

    Files are stored under ``uploads/<user_id>/`` when a user ID is
    provided, keeping each user's documents isolated on disk.
    """
    settings = get_settings()
    base_dir = Path(settings.upload_dir)

    # Store under a user-specific subdirectory when user_id is available
    if user_id:
        upload_dir = base_dir / user_id
    else:
        upload_dir = base_dir / "_anonymous"
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_filename(filename)
    dest = upload_dir / safe_name

    # Prevent file overwrite — append counter if file exists
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        counter = 1
        while dest.exists():
            dest = upload_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    # Verify the resolved path is within the upload directory
    if not dest.resolve().is_relative_to(base_dir.resolve()):
        raise ValueError("Invalid filename")

    content = file.read()
    dest.write_bytes(content if isinstance(content, bytes) else content.encode())
    logger.info("file_saved", path=str(dest), user_id=user_id, size=len(content))
    return dest


def load_document(file_path: str) -> list[Document]:
    """Load a single document and return raw LangChain Documents.

    Handles both text documents and images.
    """
    ext = Path(file_path).suffix.lower()

    # Image files → describe via vision model
    if ext in _IMAGE_EXTENSIONS:
        from app.core.preprocessing import preprocess_image
        return preprocess_image(file_path).text_docs

    loader_key = _EXTENSION_MAP.get(ext)
    if loader_key is None:
        raise ValueError(f"Unsupported file type: {ext}")
    docs = _LOADERS[loader_key](file_path)
    for doc in docs:
        doc.metadata["source"] = file_path
        doc.metadata["content_type"] = "text"
    return docs


def load_document_multimodal(file_path: str) -> tuple[list[Document], list[dict]]:
    """Load a document with full multimodal preprocessing.

    Uses the preprocessing pipeline to extract ALL visual elements
    (images, tables, flowcharts, charts, diagrams) from both PDF and
    DOCX before chunking.

    Returns ``(text_docs, image_records)`` where image_records contain
    paths and descriptions of extracted visual elements.
    """
    from app.core.preprocessing import preprocess_document

    try:
        preprocessed = preprocess_document(file_path)
    except ImportError as e:
        logger.warning("preprocessing_import_error", error=str(e))
        # Fallback: plain text-only loading
        docs = load_document(file_path)
        return docs, []
    except Exception:
        logger.exception("preprocessing_error", file=file_path)
        docs = load_document(file_path)
        return docs, []

    # Convert visual elements into image_records for the vector store
    image_records: list[dict] = []
    for ve in preprocessed.visual_elements:
        image_records.append({
            "path": ve.image_path,
            "page": ve.page,
            "description": ve.description,
            "content_type": ve.content_type,
        })

    logger.info(
        "multimodal_load_complete",
        file=file_path,
        text_docs=len(preprocessed.text_docs),
        visual_elements=len(preprocessed.visual_elements),
    )
    return preprocessed.text_docs, image_records


# ── Token counter ─────────────────────────────────────────────────────

_TIKTOKEN_ENC: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        _TIKTOKEN_ENC = tiktoken.encoding_for_model("gpt-4")  # cl100k_base
    return _TIKTOKEN_ENC


def _token_len(text: str) -> int:
    return len(_get_encoder().encode(text))


# ── Step 1: Text cleaning ────────────────────────────────────────────

_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[^\S\n]{2,}")


def _clean_text(text: str) -> str:
    """Normalise whitespace, collapse excessive blank lines, strip noise."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


# ── Step 2: Structural splitting (headers / sections) ────────────────

_MD_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


def _structural_split(docs: list[Document]) -> list[Document]:
    """Split by Markdown-style headers so section boundaries are respected.

    Also works for documents whose headers happen to start with ``#``
    (common in PDFs converted to text via Unstructured / PyPDF).
    If no headers are found the doc passes through unchanged.
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MD_HEADERS,
        strip_headers=False,  # keep headers in the chunk text
    )
    result: list[Document] = []
    for doc in docs:
        splits = md_splitter.split_text(doc.page_content)
        if not splits:
            result.append(doc)
            continue
        for split in splits:
            merged_meta = {**doc.metadata, **split.metadata}
            # Build a "section_header" breadcrumb from detected headers
            header_parts = [
                split.metadata[k] for k in ("h1", "h2", "h3", "h4") if k in split.metadata
            ]
            if header_parts:
                merged_meta["section_header"] = " > ".join(header_parts)
            result.append(Document(page_content=split.page_content, metadata=merged_meta))

    logger.info("structural_split", input=len(docs), output=len(result))
    return result


# ── Step 3: Semantic splitting ────────────────────────────────────────

def _semantic_split(docs: list[Document]) -> list[Document]:
    """Embedding-based semantic split — groups consecutive sentences
    whose embeddings are similar.  A new chunk starts when cosine
    distance crosses the configured percentile threshold."""
    settings = get_settings()
    chunker = SemanticChunker(
        embeddings=get_embeddings(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=settings.semantic_threshold,
    )
    out: list[Document] = []
    for doc in docs:
        # SemanticChunker needs meaningful text; very short docs skip it
        if _token_len(doc.page_content) < 60:
            out.append(doc)
            continue
        splits = chunker.split_documents([doc])
        out.extend(splits)
    logger.info("semantic_split", input=len(docs), output=len(out))
    return out


# ── Step 4: Token-aware size enforcement ──────────────────────────────

def _recursive_split(
    docs: list[Document], max_tokens: int, overlap_tokens: int,
) -> list[Document]:
    """Split using recursive character boundaries with a **token** budget."""
    # Approximate chars-per-token ratio (cl100k_base ≈ 3.5 chars/token)
    avg_cpt = 3.5
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(max_tokens * avg_cpt),
        chunk_overlap=int(overlap_tokens * avg_cpt),
        length_function=_token_len,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    return splitter.split_documents(docs)


def _enforce_token_limit(
    docs: list[Document], max_tokens: int, overlap_tokens: int,
) -> list[Document]:
    """Re-split any chunk that exceeds the token budget."""
    ok: list[Document] = []
    oversized: list[Document] = []
    for d in docs:
        (ok if _token_len(d.page_content) <= max_tokens else oversized).append(d)
    if oversized:
        ok.extend(_recursive_split(oversized, max_tokens, overlap_tokens))
    return ok


# ── Step 5: Contextual header injection ──────────────────────────────

def _inject_context_headers(docs: list[Document]) -> list[Document]:
    """Prepend source filename + section header to each chunk so it is
    self-contained and retrieval quality improves."""
    for doc in docs:
        parts: list[str] = []
        source = doc.metadata.get("source", "")
        if source:
            parts.append(f"Document: {Path(source).name}")
        section = doc.metadata.get("section_header", "")
        if section:
            parts.append(f"Section: {section}")
        if parts:
            header = " | ".join(parts)
            doc.page_content = f"[{header}]\n{doc.page_content}"
    return docs


# ── Step 6: Parent / child hierarchy ─────────────────────────────────

def _make_parent_id(text: str) -> str:
    return hashlib.sha256(text[:512].encode()).hexdigest()[:12]


# ── Step 7: Near-duplicate removal ───────────────────────────────────

def _dedup_chunks(docs: list[Document], threshold: float = 0.9) -> list[Document]:
    """Remove near-duplicate chunks using token-set overlap ratio."""
    seen_sets: list[set[str]] = []
    unique: list[Document] = []
    for doc in docs:
        tokens = set(re.findall(r"\w+", doc.page_content.lower()))
        is_dup = False
        for existing in seen_sets:
            overlap = len(tokens & existing) / max(len(tokens | existing), 1)
            if overlap >= threshold:
                is_dup = True
                break
        if not is_dup:
            seen_sets.append(tokens)
            unique.append(doc)
    removed = len(docs) - len(unique)
    if removed:
        logger.info("dedup_chunks", removed=removed, kept=len(unique))
    return unique


# ── Public: full chunking pipeline ────────────────────────────────────

def chunk_documents(docs: list[Document]) -> tuple[list[Document], list[Document]]:
    """Run the complete best-practice hybrid chunking pipeline.

    Returns ``(parent_chunks, child_chunks)`` where every child carries
    a ``parent_id`` linking to its parent for context expansion.

    Pipeline:
      1. Clean text
      2. Structural split (headers / sections)
      3. Semantic split (embedding cosine-distance)
      4. Token-aware size enforcement → **parent chunks**
      5. Contextual header injection
      6. Fine-grained recursive split → **child chunks**
      7. Near-duplicate removal on children
    """
    settings = get_settings()

    # 1. Clean
    for doc in docs:
        doc.page_content = _clean_text(doc.page_content)

    # 2. Structural split
    sections = _structural_split(docs)

    # 3. Semantic split within each section
    semantic_chunks = _semantic_split(sections)

    # 4. Enforce parent token limit
    parents = _enforce_token_limit(
        semantic_chunks,
        max_tokens=settings.parent_chunk_tokens,
        overlap_tokens=settings.parent_overlap_tokens,
    )
    for p in parents:
        p.metadata["chunk_type"] = "parent"
        p.metadata["chunk_id"] = _make_parent_id(p.page_content)

    # 5. Contextual headers (on a copy so parent raw text stays clean)
    parents_with_headers = _inject_context_headers(
        [Document(page_content=p.page_content, metadata=dict(p.metadata)) for p in parents]
    )

    # 6. Child splits from header-enriched parents
    children: list[Document] = []
    for parent, parent_h in zip(parents, parents_with_headers):
        child_splits = _recursive_split(
            [parent_h],
            max_tokens=settings.child_chunk_tokens,
            overlap_tokens=settings.child_overlap_tokens,
        )
        for idx, child in enumerate(child_splits):
            child.metadata["chunk_type"] = "child"
            child.metadata["parent_id"] = parent.metadata["chunk_id"]
            child.metadata["child_index"] = idx
        children.extend(child_splits)

    # 7. Deduplicate children
    children = _dedup_chunks(children)

    logger.info(
        "hybrid_chunking_complete",
        input_docs=len(docs),
        sections=len(sections),
        semantic_chunks=len(semantic_chunks),
        parent_chunks=len(parents),
        child_chunks=len(children),
    )
    return parents, children
