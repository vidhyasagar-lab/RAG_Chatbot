"""A search never reads the index while an upload or delete is changing it.

Retrieval moved into a worker thread so it would stop stalling the event
loop, but uploads and deletes still write on the loop. BM25's delete path
resets the index and re-adds, and FAISS updates its index before its id map,
so a search landing mid-write could raise IndexError, ZeroDivisionError or
KeyError. One lock now covers the reads and the writes; embedding - the slow
network part - happens outside it.
"""

from __future__ import annotations

import threading

import app.core.vector_store as vs


def _blocked_while_locked(work) -> bool:
    """True if `work` waits for the index lock and finishes once it is free."""
    finished = threading.Event()
    vs.get_vector_store()
    with vs._index_lock:
        threading.Thread(target=lambda: (work(), finished.set()), daemon=True).start()
        waited = not finished.wait(0.3)
    return waited and finished.wait(10)


def test_a_search_waits_for_an_index_write():
    assert _blocked_while_locked(lambda: vs.hybrid_search("anything", k=3))


def test_a_delete_waits_for_a_search():
    assert _blocked_while_locked(lambda: vs.delete_documents_by_source("nothing-here.pdf"))


def test_embedding_happens_outside_the_lock(monkeypatch):
    """Embedding a query is a network call; holding the lock through it would
    serialise every search behind the slowest one."""
    free_during_embed = []

    class Probe:
        def embed_query(self, text):
            got = vs._index_lock.acquire(blocking=False)
            free_during_embed.append(got)
            if got:
                vs._index_lock.release()
            return [0.0] * 32

    store = vs.get_vector_store()
    monkeypatch.setattr(store, "_embed_query", Probe().embed_query)

    def from_another_thread():
        vs.hybrid_search("anything", k=3)

    t = threading.Thread(target=from_another_thread)
    t.start()
    t.join(10)

    assert free_during_embed == [True]
