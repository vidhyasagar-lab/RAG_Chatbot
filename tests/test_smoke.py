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


def test_security_headers_present(client):
    resp = client.get("/api/v1/health")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in resp.headers


def test_no_html_is_served_anywhere(client):
    """The API is headless: the old page routes must stay gone.

    These are the exact paths the Jinja/HTMX router used to own. Re-adding a
    server-rendered surface would reintroduce the template dependency this
    change removed, so it should fail loudly rather than quietly work.
    """
    for path in ("/", "/login", "/register", "/logout", "/admin",
                 "/partials/stats", "/partials/doc-history"):
        resp = client.get(path)
        assert resp.status_code == 404, f"{path} still served: {resp.status_code}"


def test_auth_endpoints_are_json(client):
    """The replacements live under /api/v1/auth and speak JSON."""
    resp = client.post("/api/v1/auth/login", json={"username": "nobody", "password": "x"})
    assert resp.status_code == 401
    assert "application/json" in resp.headers["content-type"]
