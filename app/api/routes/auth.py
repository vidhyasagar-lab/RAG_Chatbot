"""Authentication endpoints — JSON equivalents of the removed HTML form handlers.

These replace the form-post routes that used to live in ``routes/pages.py``.
The session mechanism is unchanged: a successful login or registration sets
the same signed, httponly ``user_id`` cookie that every other route already
depends on via ``require_authenticated_user``. Only the transport changed —
JSON in, JSON out, instead of a form post that rendered a template.

Error semantics are deliberately preserved from the form handlers:
brute-force lockout still precedes credential checking, and a failed login
still reports the same message whether the username exists or not, so the
endpoint does not become a username oracle.
"""

from __future__ import annotations

from fastapi import APIRouter, Cookie, HTTPException, Response

from app.core.auth import (
    check_login_allowed,
    clear_failed_logins,
    clear_session_cookie,
    get_current_user_id,
    record_failed_login,
    set_session_cookie,
)
from app.core.logging import get_logger
from app.core.user_store import authenticate_user, get_user, register_user
from app.models.schemas import AuthCredentials, AuthUserResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


def _public_user(user: dict) -> AuthUserResponse:
    """Project a user row onto the fields safe to return to a client."""
    return AuthUserResponse(
        user_id=user["user_id"],
        username=user["username"],
        role=user.get("role", "user"),
        created_at=user.get("created_at", ""),
    )


@router.post("/login", response_model=AuthUserResponse)
async def login(credentials: AuthCredentials, response: Response) -> AuthUserResponse:
    """Verify credentials and issue a session cookie."""
    username = credentials.username.strip()
    if not username:
        raise HTTPException(status_code=422, detail="Username is required")

    # Lockout is checked before authentication, so a locked account cannot be
    # probed for password correctness by timing or response differences.
    if not check_login_allowed(username):
        logger.warning("login_locked_out", username=username)
        raise HTTPException(
            status_code=429,
            detail="Too many failed attempts. Try again in 5 minutes.",
        )

    user = authenticate_user(username, credentials.password)
    if not user:
        record_failed_login(username)
        logger.warning("login_failed", username=username)
        # Same message for "no such user" and "wrong password" — distinguishing
        # them would turn this endpoint into a username oracle.
        raise HTTPException(status_code=401, detail="Invalid username or password")

    clear_failed_logins(username)
    set_session_cookie(response, user["user_id"])
    return _public_user(user)


@router.post("/register", response_model=AuthUserResponse, status_code=201)
async def register(credentials: AuthCredentials, response: Response) -> AuthUserResponse:
    """Create a user and issue a session cookie.

    Validation (empty username, password length, name collision) lives in
    ``register_user`` and surfaces as ValueError; it is translated to 400 here
    rather than duplicated, so the rules cannot drift between call sites.
    """
    try:
        user = register_user(credentials.username, credentials.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    set_session_cookie(response, user["user_id"])
    return _public_user(user)


@router.post("/logout", status_code=204)
async def logout(response: Response) -> Response:
    """Clear the session cookie.

    POST rather than GET: the form-based version was a GET, which meant any
    page could log a user out with an <img> tag. Nothing renders HTML here
    any more, so the safer verb costs nothing.
    """
    clear_session_cookie(response)
    response.status_code = 204
    return response


@router.get("/me", response_model=AuthUserResponse)
async def me(user_id: str = Cookie(None)) -> AuthUserResponse:
    """Return the signed-in user, or 401.

    Replaces what ``GET /`` did for the server-rendered app: decide whether
    to show the login screen or the chat screen. A client calls this on load
    to resolve the same question.
    """
    uid = get_current_user_id(user_id)
    if not uid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = get_user(uid)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return _public_user(user)
