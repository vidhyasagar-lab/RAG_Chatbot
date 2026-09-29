"""Sending the sign-in code by email.

Plain ``smtplib`` rather than a provider SDK. Every transactional mail
service speaks SMTP, so switching from Brevo to SES to a Gmail app password
is a change to ``.env`` and nothing in this file — and it adds no dependency
to a project that already pulls in enough.

Two deliberate behaviours when SMTP is not configured:

* In development the code is written to the log instead of sent, so the whole
  sign-in flow can be exercised without a provider account.
* In production the caller gets an error. Logging a sign-in code to a
  production log where it cannot be read by the person signing in, while
  telling them mail is on its way, would be worse than failing.
"""

from __future__ import annotations

import smtplib
from email.message import EmailMessage
from email.utils import formataddr

from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmailNotConfigured(RuntimeError):
    """Raised when there is no way to send and no safe way to fake it."""


def _is_dev() -> bool:
    return get_settings().app_env.lower() in ("development", "dev", "local")


def _build_message(to_email: str, code: str, ttl_minutes: int) -> EmailMessage:
    """The code, as a message that survives both HTML and plain-text readers.

    The code is repeated in the subject line because many phone clients show
    the subject in the notification, which saves opening the mail at all.
    """
    settings = get_settings()
    msg = EmailMessage()
    msg["Subject"] = f"{code} is your Verity sign-in code"
    msg["From"] = formataddr((settings.mail_from_name, settings.effective_mail_from))
    msg["To"] = to_email
    # Marks this as transactional so well-behaved clients skip the "unsubscribe"
    # affordances, and so replies do not vanish into an unwatched mailbox.
    msg["Auto-Submitted"] = "auto-generated"

    msg.set_content(
        f"Your Verity sign-in code is {code}\n\n"
        f"It expires in {ttl_minutes} minutes and can be used once.\n\n"
        "If you did not ask to sign in, you can ignore this message — "
        "nobody can get in without the code.\n"
    )
    msg.add_alternative(
        f"""<!doctype html>
<html>
  <body style="margin:0;padding:32px;background:#f6f5f2;font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;color:#1c1c1a">
    <div style="max-width:440px;margin:0 auto;background:#ffffff;border:1px solid #e6e3dc;border-radius:16px;padding:32px">
      <p style="margin:0;font-size:11px;letter-spacing:0.18em;text-transform:uppercase;color:#8a877f">Verity</p>
      <h1 style="margin:16px 0 0;font-size:22px;font-weight:600;line-height:1.3">Your sign-in code</h1>
      <p style="margin:12px 0 0;font-size:14px;line-height:1.6;color:#57544c">
        Enter this code to finish signing in. It expires in {ttl_minutes} minutes and works once.
      </p>
      <p style="margin:24px 0;font-size:34px;font-weight:600;letter-spacing:0.22em;font-family:ui-monospace,SFMono-Regular,Menlo,monospace">{code}</p>
      <p style="margin:0;font-size:13px;line-height:1.6;color:#8a877f">
        If you did not ask to sign in, ignore this message. Nobody can get in without the code.
      </p>
    </div>
  </body>
</html>""",
        subtype="html",
    )
    return msg


def _send_blocking(msg: EmailMessage) -> None:
    """Hand the message to the SMTP server. Blocking; called off the loop."""
    settings = get_settings()
    # 465 is implicit TLS (SMTPS); 587 and 25 start in the clear and upgrade.
    if settings.smtp_port == 465:
        with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=15) as server:
            if settings.smtp_user:
                server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(msg)
        return

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as server:
        if settings.smtp_starttls:
            server.starttls()
        if settings.smtp_user:
            server.login(settings.smtp_user, settings.smtp_password)
        server.send_message(msg)


async def send_login_code(to_email: str, code: str, ttl_minutes: int) -> None:
    """Email a sign-in code, or log it in development.

    Raises EmailNotConfigured in production with no SMTP settings, and
    OSError/smtplib.SMTPException when the provider refuses the message. The
    caller turns either into an error the person can act on; neither is
    swallowed, because a code that was never sent must not look like one that
    was.
    """
    settings = get_settings()

    if not settings.email_sending_configured:
        if _is_dev():
            # Deliberately the one place a live code is written down. Only
            # reachable when APP_ENV is development and no SMTP host is set.
            logger.warning("login_code_not_emailed_dev_only", email=to_email, code=code)
            return
        logger.error("login_code_smtp_unconfigured", email=to_email)
        raise EmailNotConfigured(
            "SMTP is not configured. Set SMTP_HOST, SMTP_USER, SMTP_PASSWORD "
            "and MAIL_FROM in .env to send sign-in codes."
        )

    msg = _build_message(to_email, code, ttl_minutes)
    # smtplib is blocking and a slow provider would otherwise stall every
    # other request on the loop, streaming answers included.
    await run_in_threadpool(_send_blocking, msg)
    logger.info("login_code_emailed", email=to_email)
