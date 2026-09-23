"""GET /documents/figure serves the images answers cite - to their owner only.

Figures extracted from a PDF land in the shared ``uploads/extracted`` folder,
named after the document they came from; an uploaded image stays in its
owner's ``uploads/<user_id>`` folder. Ownership is decided from the database,
never from the path the client sends, so a guessed or traversing path reads as
a plain 404.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _sign_in(client) -> str:
    client.cookies.clear()
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": f"fig_{uuid.uuid4().hex[:10]}", "password": "correct-horse-battery"},
    )
    assert resp.status_code == 201, resp.text
    return client.get("/api/v1/auth/me").json()["user_id"]


def _upload_root() -> Path:
    from app.config import get_settings

    return Path(get_settings().upload_dir)


def _extracted(name: str) -> Path:
    folder = _upload_root() / "extracted"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(PNG)
    return path


def _get(client, path: Path | str):
    return client.get("/api/v1/documents/figure", params={"path": str(path)})


@pytest.fixture
def owner(client):
    """A signed-in user who owns report-<id>.pdf and one extracted figure from it."""
    from app.core.user_store import record_document

    user_id = _sign_in(client)
    stem = f"report-{uuid.uuid4().hex[:8]}"
    record_document(user_id, f"{stem}.pdf", 1000, 3, images_extracted=1)
    figure = _extracted(f"{stem}_p2_chart.png")
    yield user_id, figure
    client.cookies.clear()


def test_owner_gets_the_figure(client, owner):
    _, figure = owner
    resp = _get(client, figure)
    assert resp.status_code == 200
    assert resp.content == PNG
    assert resp.headers["content-type"] == "image/png"


def test_a_path_relative_to_the_working_directory_works_too(client, owner):
    """The loader records paths as it saved them, which with the default
    UPLOAD_DIR is relative (uploads/extracted/...)."""
    import os

    _, figure = owner
    assert _get(client, os.path.relpath(figure)).status_code == 200


def test_an_uploaded_image_in_the_owners_folder_is_served(client, owner):
    user_id, _ = owner
    folder = _upload_root() / user_id
    folder.mkdir(parents=True, exist_ok=True)
    image = folder / "whiteboard.png"
    image.write_bytes(PNG)
    assert _get(client, image).status_code == 200


def test_someone_elses_figure_is_not_found(client, owner):
    _, figure = owner
    _sign_in(client)  # a different user
    assert _get(client, figure).status_code == 404


def test_another_users_upload_folder_is_not_found(client, owner):
    user_id, _ = owner
    folder = _upload_root() / user_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "private.png").write_bytes(PNG)
    _sign_in(client)
    assert _get(client, folder / "private.png").status_code == 404


def test_a_figure_named_like_nobodys_document_is_not_found(client, owner):
    stray = _extracted(f"orphan-{uuid.uuid4().hex[:8]}_p1_table.png")
    assert _get(client, stray).status_code == 404


def test_a_path_outside_the_upload_folder_is_not_found(client, owner):
    outside = _upload_root().parent / "secret.png"
    outside.write_bytes(PNG)
    assert _get(client, outside).status_code == 404
    assert _get(client, _upload_root() / "extracted" / ".." / ".." / "secret.png").status_code == 404


def test_a_non_image_file_is_not_served(client, owner):
    user_id, _ = owner
    folder = _upload_root() / user_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "notes.txt").write_text("hello")
    assert _get(client, folder / "notes.txt").status_code == 404


def test_a_missing_file_is_not_found(client, owner):
    _, figure = owner
    assert _get(client, figure.with_name("gone.png")).status_code == 404


def test_signed_out_is_rejected(client, owner):
    _, figure = owner
    client.cookies.clear()
    assert _get(client, figure).status_code == 401
