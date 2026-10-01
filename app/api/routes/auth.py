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

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response

from app.core.auth import (
    check_login_allowed,
    clear_failed_logins,
    clear_session_cookie,
    get_current_user_id,
    lockout_remaining,
    record_failed_login,
    require_authenticated_user,
    set_session_cookie,
)
from app.config import get_settings
from app.core.login_codes import (
    CodeRequestRefused,
    discard_code,
    issue_code,
    normalise_email,
    verify_code,
)
from app.core.logging import get_logger
from app.core.mailer import EmailNotConfigured, send_login_code
from app.core.quota import usage_for
from app.core.user_store import (
    authenticate_user,
    get_or_create_user_by_email,
    get_user_documents,
    register_user,
    revoke_sessions,
)
from app.models.schemas import (
    AuthCredentials,
    AuthUserResponse,
    EmailCodeRequest,
    EmailCodeSent,
    RegisterCredentials,
    EmailCodeVerify,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


def _public_user(user: dict) -> AuthUserResponse:
    """Project a user row onto the fields safe to return to a client.

    Includes the quota so the client can show what is left, and disable the
    composer or the upload button before someone runs into a 403.
    """
    held = len(get_user_documents(user["user_id"]))
    return AuthUserResponse(
        user_id=user["user_id"],
        username=user["username"],
        role=user.get("role", "user"),
        created_at=user.get("created_at", ""),
        **usage_for(user, held=held),
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
            headers={"Retry-After": str(lockout_remaining(username))},
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
async def register(credentials: RegisterCredentials, response: Response) -> AuthUserResponse:
    """Create a user, once the address has been proved, and sign them in.

    The code is redeemed first, so a refused registration writes nothing.
    Without it this endpoint created an account from nothing but a request
    body, and it bypasses the API key - so anyone able to reach the host
    could mint accounts, each with a lifetime answer budget behind it.

    Validation (empty username, password length, name collision) lives in
    ``register_user`` and surfaces as ValueError; it is translated to 400 here
    rather than duplicated, so the rules cannot drift between call sites.
    """
    email = normalise_email(credentials.username)

    # The same lockout and attempt cap as the sign-in door, keyed by address:
    # both redeem the same code, so registering must not be a way around the
    # cap that protects signing in.
    if not check_login_allowed(email):
        raise HTTPException(
            status_code=429,
            detail="Too many failed attempts. Try again in 5 minutes.",
            headers={"Retry-After": str(lockout_remaining(email))},
        )
    if not verify_code(email, credentials.code):
        record_failed_login(email)
        logger.warning("register_code_rejected", email=email)
        raise HTTPException(status_code=401, detail="That code is wrong or has expired.")

    try:
        user = register_user(email, credentials.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    clear_failed_logins(email)
    logger.info("user_registered_verified", email=email)
    set_session_cookie(response, user["user_id"])
    return _public_user(user)


@router.post("/code/request", response_model=EmailCodeSent)
async def request_code(body: EmailCodeRequest) -> EmailCodeSent:
    """Email a one-time sign-in code to the address given.

    The same answer comes back whether or not the address has an account.
    Saying "no such user" here would let anyone check who has signed up, and
    the flow does not need the distinction: a code verified for an unknown
    address creates the account.

    A send that fails gives the code's slot back, so a provider outage does
    not leave the person inside the resend cooldown with nothing to type.
    """
    email = normalise_email(body.email)
    settings = get_settings()

    try:
        code, ttl_minutes = issue_code(email)
    except CodeRequestRefused as e:
        raise HTTPException(
            status_code=429,
            detail=str(e),
            headers={"Retry-After": str(e.retry_after)} if e.retry_after else None,
        ) from e

    try:
        await send_login_code(email, code, ttl_minutes)
    except EmailNotConfigured as e:
        discard_code(email, refund=True)
        logger.error("login_code_send_unconfigured", email=email)
        raise HTTPException(
            status_code=503,
            detail="Sign-in codes are not available right now. Use your password instead.",
        ) from e
    except Exception as e:
        discard_code(email, refund=True)
        logger.exception("login_code_send_failed", email=email)
        raise HTTPException(
            status_code=502,
            detail="That code could not be sent. Check the address, or try again in a moment.",
        ) from e

    return EmailCodeSent(
        sent=True,
        # False only in development with no SMTP configured, where the code
        # went to the server log. The client says so rather than telling
        # someone to check an inbox nothing was sent to.
        delivered=settings.email_sending_configured,
        expires_in_minutes=ttl_minutes,
        resend_in_seconds=settings.login_code_resend_seconds,
    )


@router.post("/code/verify", response_model=AuthUserResponse)
async def verify_code_and_sign_in(body: EmailCodeVerify, response: Response) -> AuthUserResponse:
    """Redeem a code, then sign in — creating the account if it is new.

    Reaching a correct code is proof the caller reads that mailbox, which is
    the whole basis for the session handed out here, so there is no password
    step and no separate registration call.
    """
    email = normalise_email(body.email)

    # The same lockout that guards passwords, keyed by address. The attempt
    # cap inside the code store kills one code; this stops someone burning
    # through a fresh code every cooldown to keep guessing.
    if not check_login_allowed(email):
        raise HTTPException(
            status_code=429,
            detail="Too many failed attempts. Try again in 5 minutes.",
            headers={"Retry-After": str(lockout_remaining(email))},
        )

    if not verify_code(email, body.code):
        record_failed_login(email)
        raise HTTPException(status_code=401, detail="That code is wrong or has expired.")

    clear_failed_logins(email)
    user, created = get_or_create_user_by_email(email)
    logger.info("signed_in_with_code", user_id=user["user_id"], created=created)
    set_session_cookie(response, user["user_id"])
    if created:
        response.status_code = 201
    return _public_user(user)


@router.post("/logout", status_code=204)
async def logout(response: Response, user_id: str = Cookie(None)) -> Response:
    """End the session: retire the cookie, then clear it.

    Clearing alone only made the browser forget the value. Anything that had
    already captured it - a shared machine, a proxy log, a backup - kept
    working until the idle window lapsed, because the cookie was a bearer
    token with nothing able to retire it. Bumping the account's session
    generation invalidates every cookie issued before this moment.

    Still answers 204 when there is no valid session: signing out of nothing
    is not an error, and saying so would report whether a cookie was good.

    POST rather than GET: the form-based version was a GET, which meant any
    page could log a user out with an <img> tag. Nothing renders HTML here
    any more, so the safer verb costs nothing.
    """
    uid = get_current_user_id(user_id)
    if uid:
        revoke_sessions(uid)
    clear_session_cookie(response)
    response.status_code = 204
    return response


@router.get("/me", response_model=AuthUserResponse)
async def me(
    current_user: dict = Depends(require_authenticated_user),
) -> AuthUserResponse:
    """Return the signed-in user, or 401.

    Replaces what ``GET /`` did for the server-rendered app: decide whether
    to show the login screen or the chat screen. A client calls this on load
    to resolve the same question.

    Uses the shared dependency rather than re-deriving the user from the
    cookie. The hand-rolled version behaved identically, but it was invisible
    to any audit that enumerates route dependencies to find unguarded
    endpoints, and it would not inherit future hardening of
    require_authenticated_user (session revocation, for instance).
    """
    return _public_user(current_user)
