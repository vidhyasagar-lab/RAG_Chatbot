"""Prove the SMTP settings in .env actually send, before trusting sign-in to them.

    uv run python -m app.scripts.send_test_email you@example.com

Sends one message shaped exactly like a real sign-in code, so a success here
means the sign-in flow will work. Nothing is printed that should not be: the
password is never shown, and the settings are reported by name only.

Common refusals are translated, because the raw SMTP error is rarely the
thing you need to change.
"""

from __future__ import annotations

import argparse
import asyncio
import smtplib
import ssl
import sys

from app.config import get_settings
from app.core.mailer import send_login_code


def _describe(settings) -> None:
    """What is configured, without revealing the credential."""
    print("Settings in use:")
    print(f"  SMTP_HOST      {settings.smtp_host or '(empty)'}")
    print(f"  SMTP_PORT      {settings.smtp_port}")
    print(f"  SMTP_STARTTLS  {settings.smtp_starttls}")
    print(f"  SMTP_USER      {settings.smtp_user or '(empty)'}")
    print(f"  SMTP_PASSWORD  {'set, ' + str(len(settings.smtp_password)) + ' characters' if settings.smtp_password else '(empty)'}")
    print(f"  MAIL_FROM      {settings.effective_mail_from or '(empty)'}")
    print(f"  APP_ENV        {settings.app_env}")
    print()


def _explain(error: Exception, settings) -> str:
    """Turn the provider's refusal into the thing to change."""
    text = str(error)

    if isinstance(error, smtplib.SMTPAuthenticationError):
        if "gmail" in settings.smtp_host.lower():
            return (
                "Gmail rejected the credentials. Almost always one of:\n"
                "  - SMTP_PASSWORD is your normal Google password. It must be a\n"
                "    16-character App Password from myaccount.google.com/apppasswords\n"
                "  - 2-Step Verification is off, so app passwords do not exist yet\n"
                "  - SMTP_USER is not the full address (needs the @gmail.com)"
            )
        return (
            "The provider rejected the credentials. Check SMTP_USER and\n"
            "SMTP_PASSWORD. On Brevo the password is the SMTP key, not your\n"
            "account password, and the user is the login Brevo shows you."
        )

    if isinstance(error, smtplib.SMTPSenderRefused):
        return (
            f"The provider will not send as {settings.effective_mail_from}.\n"
            "Verify that address with the provider first, or set MAIL_FROM to\n"
            "one you have already verified. On Gmail it must be the account's\n"
            "own address."
        )

    if isinstance(error, smtplib.SMTPRecipientsRefused):
        return "The provider refused the recipient. Check the address you passed."

    if isinstance(error, (TimeoutError, OSError)) and "10060" in text or "timed out" in text.lower():
        return (
            f"Nothing answered at {settings.smtp_host}:{settings.smtp_port}.\n"
            "Check the host and port, and whether outbound SMTP is blocked here.\n"
            "Port 587 with STARTTLS is the usual pair; 465 means implicit TLS."
        )

    if isinstance(error, ssl.SSLError) or "wrong version number" in text.lower():
        return (
            "TLS handshake failed, which usually means the port and mode disagree:\n"
            "  port 587 -> SMTP_STARTTLS=true\n"
            "  port 465 -> SMTP_STARTTLS is ignored, implicit TLS is used"
        )

    return "Unrecognised failure. The raw error is above."


def main() -> int:
    parser = argparse.ArgumentParser(description="Send one test sign-in code.")
    parser.add_argument("recipient", help="Where to send the test message.")
    parser.add_argument("--code", default="123456", help="The code to put in it.")
    args = parser.parse_args()

    settings = get_settings()
    _describe(settings)

    if not settings.email_sending_configured:
        print("SMTP is not configured: SMTP_HOST and MAIL_FROM are both needed.")
        print("Until they are set, sign-in codes are logged in development and")
        print("refused in production. Fill them in .env and run this again.")
        return 1

    print(f"Sending to {args.recipient} ...")
    try:
        asyncio.run(send_login_code(args.recipient, args.code, settings.login_code_ttl_minutes))
    except Exception as e:  # noqa: BLE001 - the point is to report anything
        print(f"\nFAILED: {type(e).__name__}: {e}\n")
        print(_explain(e, settings))
        return 1

    print("\nSent. Check that inbox, and the spam folder.")
    print(f"The message should show the code {args.code} in its subject line.")
    print("If it arrived, sign-in by code will work.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
