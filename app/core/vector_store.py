"""Hybrid vector-store: dense (FAISS) + sparse (BM25) retrieval.

The two result lists are merged with **Reciprocal Rank Fusion (RRF)**
so that documents appearing in both lists are boosted.
"""

from __future__ import annotations

import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from threading import Lock

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import get_settings
from app.core.embeddings import get_embeddings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Dense store (FAISS) ──────────────────────────────────────────────

_store: FAISS | None = None
_lock = Lock()

# ── Parent chunk cache (id → Document) ───────────────────────────────
_parent_store: dict[str, Document] = {}
# ── Image metadata cache (source → list of image records) ────────
_image_store: dict[str, list[dict]] = {}

_FAISS_INDEX_NAME = "rag_documents"
#: Sidecar holding everything FAISS does not: parent chunks, image records,
#: and the BM25 corpus. Without it these were rebuilt as empty on every
#: restart, silently collapsing hybrid retrieval to dense-only.
_STATE_FILE = "rag_documents_state.pkl"


def get_vector_store() -> FAISS:
    """Return (or create/load) the singleton FAISS vector store."""
    global _store
    if _store is not None:
        return _store

    with _lock:
        if _store is not None:
            return _store

        settings = get_settings()
        persist_dir = settings.vectorstore_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        index_path = Path(persist_dir) / f"{_FAISS_INDEX_NAME}.faiss"
        embeddings = get_embeddings()

        if index_path.exists():
            _store = FAISS.load_local(
                persist_dir,
                embeddings,
                index_name=_FAISS_INDEX_NAME,
                allow_dangerous_deserialization=True,
            )
            logger.info("faiss_store_loaded", persist_dir=persist_dir)
            _load_state()
        else:
            # Create an empty FAISS index with a dummy doc
            _store = FAISS.from_documents(
                [Document(page_content="__init__", metadata={"_placeholder": True})],
                embeddings,
            )
            logger.info("faiss_store_created", persist_dir=persist_dir)
            _save_store()

        return _store


def _state_path() -> Path:
    return Path(get_settings().vectorstore_dir) / _STATE_FILE


def _save_state() -> None:
    """Persist parent chunks, image records, and the BM25 corpus.

    The BM25 *index* is not written — its term statistics are derived, so only
    the corpus is stored and the index is rebuilt on load. Written to a temp
    file and renamed so a crash mid-write cannot leave a truncated sidecar.
    """
    try:
        path = _state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(
                {
                    "version": 1,
                    "parents": _parent_store,
                    "images": _image_store,
                    "bm25_docs": _bm25.docs,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp.replace(path)
    except Exception:
        # Never let a persistence failure break an upload that already
        # succeeded in FAISS — the sidecar rebuilds on the next save.
        logger.exception("vector_state_save_failed")


def _load_state() -> None:
    """Restore the sidecar written by :func:`_save_state`."""
    global _parent_store, _image_store, _bm25

    path = _state_path()
    if not path.exists():
        # An index written before the sidecar existed. The child chunks are
        # still in the FAISS docstore, so the sparse index can be rebuilt from
        # them — parent chunks and image records are unrecoverable, since they
        # were never written anywhere.
        logger.info("vector_state_absent", path=str(path))
        _rebuild_bm25_from_docstore()
        return

    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
    except Exception:
        logger.exception("vector_state_load_failed", path=str(path))
        return

    _parent_store = data.get("parents") or {}
    _image_store = data.get("images") or {}

    rebuilt = BM25Index()
    corpus = data.get("bm25_docs") or []
    if corpus:
        rebuilt.add_documents(corpus)
    _bm25 = rebuilt

    logger.info(
        "vector_state_loaded",
        parents=len(_parent_store),
        image_sources=len(_image_store),
        bm25_docs=len(_bm25.docs),
    )


def _rebuild_bm25_from_docstore() -> None:
    """Recover the sparse index from the dense store's documents.

    Used when an index predates the sidecar. FAISS keeps the full child
    documents, so BM25 can be reconstructed exactly; without this, hybrid
    retrieval would stay dense-only until every document was re-uploaded.
    """
    global _bm25

    if _store is None:
        return
    try:
        docs = [
            doc for doc in _store.docstore._dict.values()
            if not doc.metadata.get("_placeholder")
        ]
    except Exception:
        logger.exception("bm25_rebuild_failed")
        return

    if not docs:
        return

    rebuilt = BM25Index()
    rebuilt.add_documents(docs)
    _bm25 = rebuilt
    logger.info("bm25_rebuilt_from_docstore", docs=len(docs))
    # Write the sidecar so the next start loads it directly.
    _save_state()


def _save_store() -> None:
    """Persist the FAISS index and its sidecar state to disk."""
    if _store is None:
        return
    settings = get_settings()
    _store.save_local(settings.vectorstore_dir, index_name=_FAISS_INDEX_NAME)
    _save_state()


# ── BM25 sparse index (in-memory) ────────────────────────────────────

_TOKEN_RE = re.compile(r"\w+")


class BM25Index:
    """Minimal in-memory BM25 (Okapi) index over document chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[Document] = []
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.doc_lens: list[int] = []
        self.avg_dl: float = 0.0
        self.token_lists: list[list[str]] = []

    def _tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def add_documents(self, docs: list[Document]) -> None:
        for doc in docs:
            tokens = self._tokenize(doc.page_content)
            self.docs.append(doc)
            self.token_lists.append(tokens)
            self.doc_lens.append(len(tokens))
            seen: set[str] = set()
            for t in tokens:
                if t not in seen:
                    self.doc_freqs[t] += 1
                    seen.add(t)
        total = sum(self.doc_lens)
        self.avg_dl = total / len(self.doc_lens) if self.doc_lens else 1.0

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        query_tokens = self._tokenize(query)
        n = len(self.docs)
        if n == 0:
            return []
        scores: list[float] = [0.0] * n

        for qt in query_tokens:
            df = self.doc_freqs.get(qt, 0)
            if df == 0:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for idx, tokens in enumerate(self.token_lists):
                tf = tokens.count(qt)
                dl = self.doc_lens[idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                scores[idx] += idf * numerator / denominator

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self.docs[i], s) for i, s in ranked[:k] if s > 0]


_bm25: BM25Index = BM25Index()
_bm25_lock = Lock()


def _get_bm25() -> BM25Index:
    return _bm25


# ── Public API ────────────────────────────────────────────────────────

def add_documents(
    child_docs: list[Document],
    parent_docs: list[Document] | None = None,
    image_records: list[dict] | None = None,
    user_id: str = "",
) -> list[str]:
    """Index child chunks into both dense + sparse stores.

    Optionally stores parent chunks in memory for context expansion
    and image records for multimodal retrieval.
    """
    # Tag every child doc with user_id
    if user_id:
        for doc in child_docs:
            doc.metadata["user_id"] = user_id

    # Dense index (FAISS)
    store = get_vector_store()
    ids = store.add_documents(child_docs)

    # Sparse index (BM25)
    with _bm25_lock:
        _get_bm25().add_documents(child_docs)

    # Parent chunks, kept for context expansion at retrieval time
    if parent_docs:
        for pdoc in parent_docs:
            cid = pdoc.metadata.get("chunk_id", "")
            if cid:
                _parent_store[cid] = pdoc

    # Image records by source
    if image_records:
        for rec in image_records:
            source = rec.get("path", "")
            if source:
                _image_store.setdefault(source, []).append(rec)

    # Persist last, once every structure is updated — saving earlier would
    # write a sidecar missing this document's BM25 entry and parent chunks.
    _save_store()

    logger.info(
        "documents_indexed",
        children=len(ids),
        parents=len(parent_docs) if parent_docs else 0,
        images=len(image_records) if image_records else 0,
    )
    return ids


# ── Hybrid retrieval ──────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    weights: list[float],
    k_rrf: int = 60,
) -> list[Document]:
    """Merge multiple ranked lists using weighted RRF.

    score(d) = Σ  weight_i / (k_rrf + rank_i(d))
    """
    doc_scores: dict[int, float] = defaultdict(float)
    doc_map: dict[int, Document] = {}

    for ranked, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(ranked, start=1):
            doc_key = id(doc)
            # De-duplicate by page_content hash when different objects
            content_hash = hash(doc.page_content)
            # use content hash for fusion to properly merge across lists
            doc_scores[content_hash] += weight / (k_rrf + rank)
            if content_hash not in doc_map:
                doc_map[content_hash] = doc

    sorted_hashes = sorted(doc_scores, key=doc_scores.get, reverse=True)  # type: ignore[arg-type]
    return [doc_map[h] for h in sorted_hashes]


def hybrid_search(
    query: str,
    k: int | None = None,
    expand_parents: bool = True,
    user_id: str = "",
) -> list[Document]:
    """Run dense + sparse search, fuse with RRF, optionally expand to parents.

    When *user_id* is provided, only documents belonging to that user are returned.
    """
    settings = get_settings()
    k = k or settings.top_k_results
    fetch_k = k * 3  # over-fetch before fusion

    # 1. Dense retrieval (embedding similarity)
    store = get_vector_store()
    dense_results = store.similarity_search(query, k=fetch_k)

    # 2. Sparse retrieval (BM25 keyword)
    bm25 = _get_bm25()
    sparse_raw = bm25.search(query, k=fetch_k)
    sparse_results = [doc for doc, _score in sparse_raw]

    # Filter by user_id if provided
    if user_id:
        dense_results = [d for d in dense_results if d.metadata.get("user_id") == user_id]
        sparse_results = [d for d in sparse_results if d.metadata.get("user_id") == user_id]

    # 3. Reciprocal Rank Fusion
    fused = _reciprocal_rank_fusion(
        ranked_lists=[dense_results, sparse_results],
        weights=[settings.dense_weight, settings.sparse_weight],
    )[:k]

    # 4. Parent expansion — replace child with parent for richer context
    if expand_parents:
        expanded: list[Document] = []
        seen_parents: set[str] = set()
        for doc in fused:
            pid = doc.metadata.get("parent_id", "")
            if pid and pid in _parent_store and pid not in seen_parents:
                parent = _parent_store[pid]
                # carry child metadata for source attribution
                parent.metadata.setdefault("source", doc.metadata.get("source", ""))
                parent.metadata.setdefault("page", doc.metadata.get("page", ""))
                expanded.append(parent)
                seen_parents.add(pid)
            else:
                expanded.append(doc)
        fused = expanded[:k]

    logger.info(
        "hybrid_search",
        query_len=len(query),
        dense=len(dense_results),
        sparse=len(sparse_results),
        fused=len(fused),
    )
    return fused


# back-compat alias
def similarity_search(query: str, k: int | None = None) -> list[Document]:
    """Legacy — now delegates to hybrid_search."""
    return hybrid_search(query, k=k)


def get_collection_stats() -> dict:
    """Return basic stats about the vector store collection."""
    store = get_vector_store()
    total_images = sum(len(v) for v in _image_store.values())
    return {
        "total_documents": store.index.ntotal,
        "collection_name": _FAISS_INDEX_NAME,
        "bm25_indexed": len(_bm25.docs),
        "parent_chunks_cached": len(_parent_store),
        "images_indexed": total_images,
    }


def get_image_store() -> dict[str, list[dict]]:
    """Return the image metadata cache."""
    return _image_store


def delete_documents_by_source(filename: str, user_id: str = "") -> int:
    """Delete all chunks (FAISS + BM25 + parent + image caches) whose source ends with *filename*.

    Returns the number of FAISS vectors removed.
    """
    store = get_vector_store()

    # 1. Find matching FAISS doc IDs
    ids_to_delete: list[str] = []
    for doc_id, doc in store.docstore._dict.items():
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        src = meta.get("source", "")
        uid = meta.get("user_id", "")
        if not src.endswith(filename):
            continue
        if user_id and uid != user_id:
            continue
        ids_to_delete.append(doc_id)

    if ids_to_delete:
        store.delete(ids_to_delete)

    # 2. Rebuild BM25 without the deleted docs
    with _bm25_lock:
        remaining = [
            d for d in _bm25.docs
            if not (d.metadata.get("source", "").endswith(filename)
                    and (not user_id or d.metadata.get("user_id", "") == user_id))
        ]
        _bm25.__init__()  # reset
        if remaining:
            _bm25.add_documents(remaining)

    # 3. Remove matching parent chunks
    parent_ids_to_remove = [
        cid for cid, pdoc in _parent_store.items()
        if pdoc.metadata.get("source", "").endswith(filename)
    ]
    for cid in parent_ids_to_remove:
        _parent_store.pop(cid, None)

    # 4. Remove matching image records
    img_keys_to_remove = [
        key for key in _image_store if key.endswith(filename)
    ]
    for key in img_keys_to_remove:
        _image_store.pop(key, None)

    # Persist after every structure has been pruned, so a restart cannot
    # resurrect the deleted document's BM25 entries or parent chunks.
    _save_store()

    logger.info("documents_deleted_by_source", filename=filename, user_id=user_id, removed=len(ids_to_delete))
    return len(ids_to_delete)
