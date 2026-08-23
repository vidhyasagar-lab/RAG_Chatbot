"""Shared test fixtures.

Every test runs against a throwaway data directory and a fake embedding
model, so the suite never touches real user data and never calls Azure.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Environment must be set before any app module is imported, because
# Settings is read at import time in several places.
_TMP = Path(tempfile.mkdtemp(prefix="ragtests_"))
os.environ.update(
    AZURE_OPENAI_API_KEY="test-key",
    AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com",
    SECRET_KEY="test-secret-not-the-default-value",
    APP_ENV="development",
    LANGFUSE_ENABLED="false",
    API_KEY="",
    VECTORSTORE_DIR=str(_TMP / "vectorstore"),
    DATA_DIR=str(_TMP / "data"),
    UPLOAD_DIR=str(_TMP / "uploads"),
    # Effectively unlimited for the shared client: the limiter buckets every
    # test under the same "testclient" address, so a real budget here would
    # make unrelated tests fail depending on execution order. Limiting itself
    # is tested in isolation against a purpose-built app.
    RATE_LIMIT="100000/minute",
)


from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Deterministic offline stand-in for AzureOpenAIEmbeddings.

    Must subclass ``Embeddings``: FAISS treats anything else as a plain
    callable and invokes it directly.
    """

    def _vec(self, text: str) -> list[float]:
        import random

        h = 0
        for ch in text:
            h = (h * 31 + ord(ch)) % 100003
        rnd = random.Random(h)
        return [rnd.random() for _ in range(32)]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture(scope="session", autouse=True)
def _stub_embeddings():
    """Replace the embedding model everywhere it is looked up."""
    import app.core.embeddings as embeddings_mod
    import app.core.vector_store as vector_store_mod

    embeddings_mod.get_embeddings = lambda: FakeEmbeddings()
    vector_store_mod.get_embeddings = lambda: FakeEmbeddings()
    yield


@pytest.fixture(scope="session")
def client(_stub_embeddings):
    """A TestClient with the app lifespan run (startup + shutdown)."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def registered_user(client):
    """Register a fresh user and return (username, password)."""
    import uuid

    username = f"user_{uuid.uuid4().hex[:10]}"
    password = "correct-horse-battery"
    resp = client.post(
        "/register",
        data={"username": username, "password": password},
        follow_redirects=False,
    )
    assert resp.status_code == 200, resp.text
    return username, password
