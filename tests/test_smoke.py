"""Smoke tests: the app boots and its core surfaces respond."""

from __future__ import annotations


def test_app_imports_and_builds():
    from app.main import app

    assert app.title == "RAG Chatbot API"


def test_health_returns_200(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_openapi_schema_builds(client):
    """Exercises every request/response model in one call."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert len(paths) > 20, f"only {len(paths)} paths registered"


def test_login_page_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_security_headers_present(client):
    resp = client.get("/api/v1/health")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in resp.headers


def test_htmx_partials_render(client):
    for path in ("/partials/stats", "/partials/doc-history"):
        resp = client.get(path)
        assert resp.status_code == 200, f"{path} -> {resp.status_code}"
