"""Per-user budgets for answers and uploads.

Three endpoints enforce these - POST /chat/, POST /chat/stream and
POST /documents/upload - so the rule lives here rather than being written out
three times and drifting apart.

Two different kinds of limit:

* **Exchanges** are a lifetime cap. The count lives in ``users.exchanges_used``
  and never resets. It cannot be derived by counting ``chat_messages``, because
  ``delete_session`` deletes a session's messages: a derived count would hand
  the user a fresh budget every time they cleared their history.
* **Documents** cap what a user currently holds, so deleting one frees the
  slot. That bounds disk and index size, not ingest spend - a user can cycle
  documents indefinitely. Deliberate: losing a slot to a misclick is worse
  than the spend it would save.

Both refuse with 403. 429 already means two other things in this API - the
rate limiter and the login lockout - and both of those clear on their own,
where a lifetime quota never does.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def is_exempt(user: dict[str, Any]) -> bool:
    """Admins are not metered."""
    return user.get("role") == "admin"


def check_exchange_quota(user: dict[str, Any]) -> None:
    """Raise 403 if this user has spent their lifetime answer budget.

    Called before any Azure call, so exceeding the limit costs nothing.
    """
    if is_exempt(user):
        return
    limit = get_settings().max_exchanges_per_user
    used = user.get("exchanges_used") or 0
    if used >= limit:
        logger.info("exchange_quota_exhausted", user_id=user.get("user_id"), used=used)
        raise HTTPException(
            status_code=403,
            detail=(
                f"You have used all {limit} of your questions. "
                "This is a demo account with a fixed budget."
            ),
        )


def check_document_quota(user: dict[str, Any], held: int) -> None:
    """Raise 403 if this user already holds the maximum number of documents.

    ``held`` is the caller's current document count, so deleting one frees a
    slot. Called before the upload is read or saved.
    """
    if is_exempt(user):
        return
    limit = get_settings().max_documents_per_user
    if held >= limit:
        logger.info("document_quota_exhausted", user_id=user.get("user_id"), held=held)
        raise HTTPException(
            status_code=403,
            detail=(
                f"You can keep {limit} documents at a time. "
                "Delete one to upload another."
            ),
        )


def usage_for(user: dict[str, Any], held: int) -> dict[str, Any]:
    """Both budgets, for the client to display.

    An exempt user reports ``None`` limits rather than a large number, so the
    UI can say "unlimited" instead of implying a ceiling that is not there.
    """
    settings = get_settings()
    exempt = is_exempt(user)
    return {
        "exchanges_used": user.get("exchanges_used") or 0,
        "exchanges_limit": None if exempt else settings.max_exchanges_per_user,
        "documents_used": held,
        "documents_limit": None if exempt else settings.max_documents_per_user,
    }
