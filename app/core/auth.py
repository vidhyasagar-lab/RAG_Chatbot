"""Authentication utilities — signed cookies, brute force protection, user resolution."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

from fastapi import Cookie, HTTPException, Request, Response
from itsdangerous import BadSignature, URLSafeTimedSerializer

from app.config import get_settings
from app.core.logging import get_logger
from app.core.user_store import get_user

logger = get_logger(__name__)

# ── Signed cookie helpers ────────────────────────────────────────────

_COOKIE_NAME = "user_id"
_COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 7 days


def _get_signer() -> URLSafeTimedSerializer:
    settings = get_settings()
    return URLSafeTimedSerializer(settings.secret_key, salt="session-cookie")


def sign_user_id(user_id: str) -> str:
    """Return an HMAC-signed token for the given user_id."""
    return _get_signer().dumps(user_id)


def unsign_user_id(token: str, max_age: int = _COOKIE_MAX_AGE) -> str | None:
    """Verify and decode a signed token. Returns user_id or None."""
    try:
        return _get_signer().loads(token, max_age=max_age)
    except BadSignature:
        # Tampered, forged, or expired — the ordinary rejection path.
        return None
    except Exception:
        # Malformed input reaching itsdangerous internals. Still "not signed
        # in", but log it: unlike BadSignature this is not expected traffic.
        logger.exception("session_token_unreadable")
        return None


def set_session_cookie(response: Response, user_id: str) -> None:
    """Set a signed, secure session cookie on the response."""
    settings = get_settings()
    is_dev = settings.app_env.lower() in ("development", "dev", "local")
    response.set_cookie(
        key=_COOKIE_NAME,
        value=sign_user_id(user_id),
        httponly=True,
        secure=not is_dev,
        samesite="lax",
        max_age=_COOKIE_MAX_AGE,
    )


def clear_session_cookie(response: Response) -> None:
    """Delete the session cookie.

    The attributes must match those used when setting it — a browser treats
    cookies differing in ``samesite``/``secure``/``path`` as distinct and
    would otherwise leave the original in place.
    """
    settings = get_settings()
    is_dev = settings.app_env.lower() in ("development", "dev", "local")
    response.delete_cookie(
        key=_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=not is_dev,
        samesite="lax",
    )


# ── Cookie-based user resolution ────────────────────────────────────

def get_current_user_id(user_id: str = Cookie(None)) -> str | None:
    """Extract and verify user_id from signed cookie. Returns user_id or None."""
    if not user_id:
        return None
    return unsign_user_id(user_id)


def require_authenticated_user(user_id: str = Cookie(None)) -> dict[str, Any]:
    """FastAPI dependency: resolve and validate the signed session cookie.

    Returns the full user dict or raises 401.
    """
    uid = get_current_user_id(user_id)
    if not uid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = get_user(uid)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_admin_user(user_id: str = Cookie(None)) -> dict[str, Any]:
    """FastAPI dependency: require an authenticated admin user."""
    user = require_authenticated_user(user_id)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ── Brute-force protection ──────────────────────────────────────────

_MAX_ATTEMPTS = 5
_LOCKOUT_SECONDS = 300  # 5 minutes
_attempts: dict[str, list[float]] = {}  # username → list of timestamps
_lockouts: dict[str, float] = {}        # username → lockout-until timestamp
_bf_lock = Lock()


def check_login_allowed(username: str) -> bool:
    """Return True if login attempts are allowed for this username."""
    now = time.time()
    with _bf_lock:
        lockout_until = _lockouts.get(username, 0)
        if now < lockout_until:
            return False
        # Clean expired lockout
        if username in _lockouts and now >= lockout_until:
            del _lockouts[username]
    return True


def record_failed_login(username: str) -> None:
    """Record a failed login attempt. Triggers lockout after MAX_ATTEMPTS."""
    now = time.time()
    with _bf_lock:
        timestamps = _attempts.setdefault(username, [])
        # Keep only recent attempts within the lockout window
        timestamps[:] = [t for t in timestamps if now - t < _LOCKOUT_SECONDS]
        timestamps.append(now)
        if len(timestamps) >= _MAX_ATTEMPTS:
            _lockouts[username] = now + _LOCKOUT_SECONDS
            _attempts[username] = []
            logger.warning("account_locked", username=username, lockout_seconds=_LOCKOUT_SECONDS)


def clear_failed_logins(username: str) -> None:
    """Clear failed login records on successful authentication."""
    with _bf_lock:
        _attempts.pop(username, None)
        _lockouts.pop(username, None)
