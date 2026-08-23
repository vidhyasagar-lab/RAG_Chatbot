"""RET-1: parent chunks, image records and the BM25 corpus must survive restart.

Before the fix only the FAISS index was persisted, so after a restart the
sparse half of hybrid retrieval returned nothing and parent expansion stopped
working - silently, because FAISS still loaded cleanly.
"""

from __future__ import annotations

import importlib

import pytest
from langchain_core.documents import Document


def _reload_store():
    """Simulate a process restart: drop every module-level global."""
    import app.core.vector_store as vs
    from tests.conftest import FakeEmbeddings

    importlib.reload(vs)
    vs.get_embeddings = lambda: FakeEmbeddings()
    return vs


@pytest.fixture
def store(_stub_embeddings):
    vs = _reload_store()
    yield vs


def _sample_docs(source="persist.pdf", user="u-persist"):
    children = [
        Document(
            page_content="zero trust network segmentation policy",
            metadata={"source": source, "chunk_id": "c1", "user_id": user},
        ),
        Document(
            page_content="incident response runbook escalation matrix",
            metadata={"source": source, "chunk_id": "c2", "user_id": user},
        ),
    ]
    parents = [
        Document(page_content="PARENT ONE " * 10, metadata={"chunk_id": "c1", "source": source}),
        Document(page_content="PARENT TWO " * 10, metadata={"chunk_id": "c2", "source": source}),
    ]
    images = [{"path": f"uploads/{source}", "description": "network diagram", "content_type": "diagram"}]
    return children, parents, images


def test_state_survives_restart(store):
    children, parents, images = _sample_docs()
    store.add_documents(child_docs=children, parent_docs=parents,
                        image_records=images, user_id="u-persist")

    assert len(store._bm25.docs) == 2
    assert len(store._parent_store) == 2
    assert len(store._image_store) == 1
    assert store._bm25.search("segmentation policy", k=3), "keyword match before restart"

    vs = _reload_store()
    assert len(vs._bm25.docs) == 0, "reload should start empty"
    vs.get_vector_store()

    assert len(vs._bm25.docs) == 2, "BM25 corpus lost on restart"
    assert len(vs._parent_store) == 2, "parent chunks lost on restart"
    assert len(vs._image_store) == 1, "image records lost on restart"

    hits = vs._bm25.search("segmentation policy", k=3)
    assert hits, "hybrid retrieval collapsed to dense-only after restart"
    assert "segmentation" in hits[0][0].page_content


def test_sidecar_file_is_written(store):
    children, parents, images = _sample_docs(source="sidecar.pdf")
    store.add_documents(child_docs=children, parent_docs=parents,
                        image_records=images, user_id="u-sidecar")
    assert store._state_path().exists(), "no sidecar written next to the FAISS index"


def test_deletion_is_persisted(store):
    children, parents, images = _sample_docs(source="deleteme.pdf", user="u-del")
    store.add_documents(child_docs=children, parent_docs=parents,
                        image_records=images, user_id="u-del")

    removed = store.delete_documents_by_source("deleteme.pdf", user_id="u-del")
    assert removed == 2

    vs = _reload_store()
    vs.get_vector_store()
    remaining = [d for d in vs._bm25.docs if d.metadata.get("source") == "deleteme.pdf"]
    assert not remaining, "deleted document came back after restart"
    assert not [k for k in vs._image_store if k.endswith("deleteme.pdf")]


def test_bm25_rebuilt_when_sidecar_is_missing(store):
    """An index predating the sidecar must still get a working sparse half."""
    children, parents, images = _sample_docs(source="legacy.pdf", user="u-legacy")
    store.add_documents(child_docs=children, parent_docs=parents,
                        image_records=images, user_id="u-legacy")

    # Simulate an index written before the sidecar existed.
    store._state_path().unlink()

    vs = _reload_store()
    vs.get_vector_store()

    assert len(vs._bm25.docs) >= 2, "BM25 not rebuilt from the FAISS docstore"
    assert vs._bm25.search("segmentation policy", k=3), "rebuilt BM25 finds nothing"
    assert vs._state_path().exists(), "rebuild should write the sidecar for next time"


def test_corrupt_sidecar_does_not_crash_startup(store):
    children, parents, images = _sample_docs(source="corrupt.pdf")
    store.add_documents(child_docs=children, parent_docs=parents,
                        image_records=images, user_id="u-corrupt")
    store._state_path().write_bytes(b"this is not a pickle")

    vs = _reload_store()
    vs.get_vector_store()          # must not raise
    assert len(vs._bm25.docs) == 0, "corrupt sidecar should degrade to empty, not crash"
