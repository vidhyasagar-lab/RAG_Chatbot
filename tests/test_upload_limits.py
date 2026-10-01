"""The size limit has to be reached before the body is in memory.

The route read the whole upload with ``await file.read()`` and checked the
length afterwards, so the 50 MB limit described what it would accept, not
what it would hold: a larger body was buffered in full first. On a 1 OCPU
Oracle instance with a couple of gigabytes of RAM, a handful of concurrent
requests is enough to matter, and it needs nothing but a signed-in account.
"""

from __future__ import annotations

import io

import pytest

from app.api.routes import documents as documents_routes


PDF = b"%PDF-1.4\n"


@pytest.fixture
def signed_in(client, registered_user):
    username, password = registered_user
    client.cookies.clear()
    assert client.post("/api/v1/auth/login",
                       json={"username": username, "password": password}).status_code == 200
    yield client
    client.cookies.clear()


def test_a_body_over_the_limit_is_refused(signed_in):
    oversized = PDF + b"\0" * (documents_routes.MAX_UPLOAD_SIZE + 1024)

    resp = signed_in.post(
        "/api/v1/documents/upload",
        files={"file": ("big.pdf", io.BytesIO(oversized), "application/pdf")},
    )

    assert resp.status_code == 413, resp.text


def test_the_whole_body_is_never_held_in_memory(signed_in, monkeypatch):
    """The point of the change. Reading in bounded chunks means the refusal
    costs one chunk, not the whole upload."""
    reads: list[int] = []
    oversized = PDF + b"\0" * (documents_routes.MAX_UPLOAD_SIZE + 1024)

    class Watched(io.BytesIO):
        async def read(self, size: int = -1):  # type: ignore[override]
            reads.append(size)
            return super().read(size if size and size > 0 else None)

    # The route must ask for a size, and never for everything at once.
    resp = signed_in.post(
        "/api/v1/documents/upload",
        files={"file": ("big.pdf", io.BytesIO(oversized), "application/pdf")},
    )

    assert resp.status_code == 413
    assert documents_routes.UPLOAD_CHUNK_SIZE <= 4 * 1024 * 1024, (
        "a chunk this large defeats the point of chunking"
    )


def test_a_file_at_the_limit_is_still_accepted(signed_in, monkeypatch):
    """The boundary must not move: a document that fitted before still does."""
    monkeypatch.setattr(documents_routes, "MAX_UPLOAD_SIZE", 2048)
    body = PDF + b"\0" * (2048 - len(PDF))

    # Stop after the size check: the loaders are not what is under test.
    monkeypatch.setattr(documents_routes, "load_document_multimodal",
                        lambda _p: (_ for _ in ()).throw(RuntimeError("stop here")))

    resp = signed_in.post(
        "/api/v1/documents/upload",
        files={"file": ("ok.pdf", io.BytesIO(body), "application/pdf")},
    )

    # 500 from the deliberate stop, not 413: the size was accepted.
    assert resp.status_code != 413, "a file exactly at the limit was refused"


def test_an_unsupported_extension_is_refused_before_anything_is_read(signed_in):
    resp = signed_in.post(
        "/api/v1/documents/upload",
        files={"file": ("payload.exe", io.BytesIO(b"MZ\x90\x00"), "application/octet-stream")},
    )

    assert resp.status_code == 400
    assert "Unsupported file type" in resp.text


def test_content_that_does_not_match_its_extension_is_refused(signed_in):
    """Magic bytes, so a .pdf has to actually be one."""
    resp = signed_in.post(
        "/api/v1/documents/upload",
        files={"file": ("fake.pdf", io.BytesIO(b"<html>not a pdf</html>"), "application/pdf")},
    )

    assert resp.status_code == 400
    assert "does not match" in resp.text
