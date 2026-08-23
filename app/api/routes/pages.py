"""Server-side rendered pages — Jinja2 + HTMX."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Cookie, Form, Header, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.core.auth import (
    check_login_allowed,
    clear_failed_logins,
    clear_session_cookie,
    get_current_user_id,
    record_failed_login,
    set_session_cookie,
)
from app.core.logging import get_logger
from app.core.user_store import authenticate_user, get_user, get_user_documents, register_user
from app.core.vector_store import get_collection_stats

logger = get_logger(__name__)

router = APIRouter(tags=["pages"])

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))


# ── Pages ─────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def index(request: Request, user_id: str = Cookie(None)):
    """Serve login or app depending on session cookie."""
    uid = get_current_user_id(user_id)
    if uid:
        user = get_user(uid)
        if user:
            return templates.TemplateResponse(request, "app.html", {"user": user})
    return templates.TemplateResponse(request, "login.html")


@router.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    """Handle login form — verify credentials and render app."""
    username = username.strip()
    if not username:
        return templates.TemplateResponse(request, "login.html", {"error": "Username is required", "tab": "login"})

    # Brute force protection
    if not check_login_allowed(username):
        logger.warning("login_locked_out", username=username)
        return templates.TemplateResponse(request, "login.html", {"error": "Too many failed attempts. Try again in 5 minutes.", "tab": "login"})

    user = authenticate_user(username, password)
    if not user:
        record_failed_login(username)
        logger.warning("login_failed", username=username)
        return templates.TemplateResponse(request, "login.html", {"error": "Invalid username or password", "tab": "login"})

    clear_failed_logins(username)
    response = templates.TemplateResponse(request, "app.html", {"user": user})
    set_session_cookie(response, user["user_id"])
    return response


@router.post("/register", response_class=HTMLResponse)
async def register(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    """Handle registration form — create user and render app."""
    username = username.strip()
    if not username:
        return templates.TemplateResponse(request, "login.html", {"error": "Username is required", "tab": "register"})
    try:
        user = register_user(username, password)
    except ValueError as e:
        return templates.TemplateResponse(request, "login.html", {"error": str(e), "tab": "register"})
    response = templates.TemplateResponse(request, "app.html", {"user": user})
    set_session_cookie(response, user["user_id"])
    return response


@router.get("/logout")
async def logout():
    """Clear session cookie and redirect to login."""
    response = RedirectResponse(url="/", status_code=302)
    clear_session_cookie(response)
    return response


@router.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, user_id: str = Cookie(None)):
    """Serve admin dashboard — only for admin users."""
    uid = get_current_user_id(user_id)
    if not uid:
        return RedirectResponse(url="/", status_code=302)
    user = get_user(uid)
    if not user or user.get("role") != "admin":
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse(request, "admin.html", {"user": user})


# ── HTMX Partials ────────────────────────────────────────────────────

@router.get("/partials/stats", response_class=HTMLResponse)
async def partial_stats(request: Request):
    """Return sidebar stats fragment."""
    stats = get_collection_stats()
    return templates.TemplateResponse(request, "partials/stats.html", {"stats": stats})


@router.get("/partials/doc-history", response_class=HTMLResponse)
async def partial_doc_history(
    request: Request,
    user_id: str = Cookie(None),
):
    """Return document history list fragment."""
    uid = get_current_user_id(user_id) or ""
    documents = get_user_documents(uid) if uid else []
    return templates.TemplateResponse(
        request,
        "partials/doc_history.html",
        {"documents": documents},
    )
