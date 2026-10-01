"""Authentication utilities — signed cookies, brute force protection, user resolution."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

from fastapi import Cookie, HTTPException, Request, Response
from itsdangerous import BadSignature, URLSafeTimedSerializer

from app.config import get_settings
from app.core.logging import get_logger
from app.core.user_store import get_user, token_version

logger = get_logger(__name__)

# ── Signed cookie helpers ────────────────────────────────────────────

COOKIE_NAME = "user_id"
# Re-sign at most this often: the timestamp only needs minute precision, and
# re-signing on every request would put a Set-Cookie on every response.
_REFRESH_AFTER_SECONDS = 60


def idle_seconds() -> int:
    """How long a session survives without an authenticated request."""
    return max(1, get_settings().session_idle_minutes) * 60


def _get_signer() -> URLSafeTimedSerializer:
    settings = get_settings()
    return URLSafeTimedSerializer(settings.secret_key, salt="session-cookie")


def sign_user_id(user_id: str) -> str:
    """Return an HMAC-signed token naming the user and their session generation.

    The generation is what makes a cookie revocable. Signing the id alone
    made it a bearer token nothing could retire: signing out cleared the
    browser's copy and any captured copy kept working until the idle window
    lapsed.
    """
    return _get_signer().dumps([user_id, token_version(user_id) or 0])


def _read_token(token: str, max_age: int | None) -> tuple[str, int] | None:
    """The (user_id, generation) a token carries, if the signature holds."""
    try:
        payload = _get_signer().loads(
            token, max_age=idle_seconds() if max_age is None else max_age)
    except BadSignature:
        # Tampered, forged, or expired — the ordinary rejection path.
        return None
    except Exception:
        # Malformed input reaching itsdangerous internals. Still "not signed
        # in", but log it: unlike BadSignature this is not expected traffic.
        logger.exception("session_token_unreadable")
        return None

    # The old payload was a bare user id with no generation to check against.
    # Such a token cannot be shown to be current, so it is refused: everyone
    # signs in once more after this ships, which is the cost of being able to
    # revoke at all.
    if not isinstance(payload, list) or len(payload) != 2:
        logger.info("session_token_unversioned")
        return None
    user_id, version = payload
    if not isinstance(user_id, str) or not isinstance(version, int):
        return None
    return user_id, version


def unsign_user_id(token: str, max_age: int | None = None) -> str | None:
    """Verify a signed token and return its user_id, or None.

    ``max_age`` defaults to the idle timeout: the token's timestamp is when it
    was last re-signed, so this rejects a session idle for longer than that.

    The generation is checked against the account on every call, which is one
    small indexed read. A token naming a generation that is no longer current
    was issued before a sign-out or a password change and is refused.
    """
    read = _read_token(token, max_age)
    if not read:
        return None
    user_id, version = read
    if token_version(user_id) != version:
        logger.info("session_token_revoked", user_id=user_id)
        return None
    return user_id


def set_session_cookie(response: Response, user_id: str) -> None:
    """Set a signed, secure session cookie on the response."""
    settings = get_settings()
    is_dev = settings.app_env.lower() in ("development", "dev", "local")
    response.set_cookie(
        key=COOKIE_NAME,
        value=sign_user_id(user_id),
        httponly=True,
        secure=not is_dev,
        samesite="lax",
        max_age=idle_seconds(),
    )


def refreshed_session_cookie(token: str) -> str | None:
    """A Set-Cookie header value that restarts the idle window, or None.

    None when the token is invalid or expired (nothing to extend) or was
    signed under a minute ago (nothing worth extending yet).
    """
    try:
        payload, signed_at = _get_signer().loads(
            token, max_age=idle_seconds(), return_timestamp=True)
    except BadSignature:
        return None
    except Exception:
        logger.exception("session_token_unreadable")
        return None
    if time.time() - signed_at.timestamp() < _REFRESH_AFTER_SECONDS:
        return None
    # Unversioned, or from a retired generation: nothing to extend.
    if not isinstance(payload, list) or len(payload) != 2:
        return None
    user_id, version = payload
    if token_version(user_id) != version:
        return None
    response = Response()
    set_session_cookie(response, user_id)
    return response.headers["set-cookie"]


def clear_session_cookie(response: Response) -> None:
    """Delete the session cookie.

    The attributes must match those used when setting it — a browser treats
    cookies differing in ``samesite``/``secure``/``path`` as distinct and
    would otherwise leave the original in place.
    """
    settings = get_settings()
    is_dev = settings.app_env.lower() in ("development", "dev", "local")
    response.delete_cookie(
        key=COOKIE_NAME,
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
_attempts: dict[str, list[float]] = {}  # scoped key → list of timestamps
_lockouts: dict[str, float] = {}        # scoped key → lockout-until timestamp
_bf_lock = Lock()

#: The doors, counted separately.
#:
#: Both are reached by address, so a single counter had them sharing one: five
#: wrong password guesses also locked the person out of signing in by emailed
#: code - the door they would reach for precisely because they could not
#: remember the password. It also let anyone who knows an address close both
#: doors on demand.
#:
#: Deliberately not keyed per client address as well. That would fix the
#: nuisance and break the defence: a distributed attacker would get five
#: guesses per address they came from instead of five in total, which is the
#: attack this exists to stop. A five-minute wait with the other door open is
#: the better trade.
PASSWORD_SCOPE = "password"
CODE_SCOPE = "code"


def _scoped(identity: str, scope: str) -> str:
    return f"{scope}:{identity}"


def check_login_allowed(username: str, scope: str = PASSWORD_SCOPE) -> bool:
    """Return True if attempts are allowed for this identity on this door."""
    key = _scoped(username, scope)
    now = time.time()
    with _bf_lock:
        lockout_until = _lockouts.get(key, 0)
        if now < lockout_until:
            return False
        # Clean expired lockout
        if key in _lockouts and now >= lockout_until:
            del _lockouts[key]
    return True


def record_failed_login(username: str, scope: str = PASSWORD_SCOPE) -> None:
    """Record a failed attempt. Triggers lockout after MAX_ATTEMPTS."""
    key = _scoped(username, scope)
    now = time.time()
    with _bf_lock:
        timestamps = _attempts.setdefault(key, [])
        # Keep only recent attempts within the lockout window
        timestamps[:] = [t for t in timestamps if now - t < _LOCKOUT_SECONDS]
        timestamps.append(now)
        if len(timestamps) >= _MAX_ATTEMPTS:
            _lockouts[key] = now + _LOCKOUT_SECONDS
            _attempts[key] = []
            logger.warning("account_locked", username=username, scope=scope,
                           lockout_seconds=_LOCKOUT_SECONDS)


def clear_failed_logins(username: str, scope: str = PASSWORD_SCOPE) -> None:
    """Clear failed attempt records for one door on successful use of it."""
    key = _scoped(username, scope)
    with _bf_lock:
        _attempts.pop(key, None)
        _lockouts.pop(key, None)


def lockout_remaining(username: str, scope: str = PASSWORD_SCOPE) -> int:
    """Seconds until this identity may try this door again (at least 1)."""
    with _bf_lock:
        until = _lockouts.get(_scoped(username, scope), 0)
    return max(1, int(until - time.time()))
