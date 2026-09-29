"""Sign-in by emailed code: issuing, redeeming, and the limits around it.

Mail is never actually sent here. ``send_login_code`` is replaced with a
recorder, which is also how a test learns the code — nothing else exposes
it, by design.
"""

from __future__ import annotations

import time
import uuid

import pytest

from app.core import login_codes


def _address() -> str:
    return f"code_{uuid.uuid4().hex[:10]}@example.test"


@pytest.fixture
def sent(monkeypatch):
    """Capture codes instead of mailing them. Returns the list of sends."""
    import app.api.routes.auth as auth_routes

    box: list[tuple[str, str, int]] = []

    async def _record(to_email: str, code: str, ttl_minutes: int) -> None:
        box.append((to_email, code, ttl_minutes))

    monkeypatch.setattr(auth_routes, "send_login_code", _record)
    return box


def test_the_suite_cannot_reach_a_real_mail_provider():
    """A guard, not a behaviour.

    Settings reads .env as well as the environment, so a developer with
    working SMTP credentials would otherwise have them pulled into the test
    run — and the test that exercises the real sender would try to deliver a
    sign-in code to an address in example.test. conftest pins these empty;
    this fails loudly if that ever stops working.
    """
    from app.config import get_settings

    settings = get_settings()
    assert not settings.email_sending_configured, (
        "the suite picked up real mail settings; conftest must pin SMTP_HOST "
        "and MAIL_FROM empty before any app module is imported"
    )
    assert not settings.smtp_password, "a real SMTP credential reached the tests"


# ── Addresses ────────────────────────────────────────────────────────

def test_addresses_are_matched_case_insensitively():
    assert login_codes.normalise_email("  Ada@Example.TEST ") == "ada@example.test"


@pytest.mark.parametrize(
    "value",
    ["ada@example.test", "a.b+tag@sub.example.co.uk", "x@y.zz"],
)
def test_plausible_addresses_accepted(value):
    assert login_codes.looks_like_email(value)


@pytest.mark.parametrize(
    "value",
    ["", "ada", "ada@", "@example.test", "ada@example", "a b@example.test", "a@@b.test"],
)
def test_implausible_addresses_rejected(value):
    assert not login_codes.looks_like_email(value)


# ── The code itself ──────────────────────────────────────────────────

def test_a_correct_code_is_accepted_once():
    email = _address()
    code, ttl = login_codes.issue_code(email)
    assert len(code) == 6 and code.isdigit()
    assert ttl >= 1

    assert login_codes.verify_code(email, code) is True
    # Consumed: the same code must not work twice.
    assert login_codes.verify_code(email, code) is False


def test_the_code_is_not_stored_in_the_clear():
    email = _address()
    code, _ = login_codes.issue_code(email)
    row = login_codes._conn().execute(
        "SELECT code_hash FROM login_codes WHERE email = ?", (email,)
    ).fetchone()
    assert code not in row["code_hash"]
    assert len(row["code_hash"]) == 64  # sha256 hex


def test_a_code_is_bound_to_its_address():
    """A row lifted from the table cannot be replayed against another account."""
    one, two = _address(), _address()
    code, _ = login_codes.issue_code(one)
    login_codes.issue_code(two)
    assert login_codes.verify_code(two, code) is False


def test_wrong_codes_run_out_of_attempts(monkeypatch):
    from app.config import get_settings

    email = _address()
    code, _ = login_codes.issue_code(email)
    limit = get_settings().login_code_max_attempts

    for i in range(limit - 1):
        assert login_codes.verify_code(email, "000000") is False
        assert login_codes.attempts_remaining(email) == limit - (i + 1)

    # The last allowed failure destroys the code, so even the right one dies.
    assert login_codes.verify_code(email, "000000") is False
    assert login_codes.attempts_remaining(email) == 0
    assert login_codes.verify_code(email, code) is False


def test_an_expired_code_is_refused(monkeypatch):
    email = _address()
    code, _ = login_codes.issue_code(email)
    # Reach past the expiry rather than sleeping through it. The real clock is
    # captured first: the replacement lands on the same module object the
    # lambda would otherwise call, which is an infinite recursion.
    real_time = time.time
    monkeypatch.setattr(login_codes.time, "time", lambda: real_time() + 3600)
    assert login_codes.verify_code(email, code) is False
    assert login_codes.debug_state(email) is None, "expired code left behind"


def test_asking_again_replaces_the_previous_code(monkeypatch):
    """An older mail in the inbox stops working once a newer one arrives."""
    monkeypatch.setattr(
        login_codes.get_settings(), "login_code_resend_seconds", 0, raising=False
    )
    email = _address()
    first, _ = login_codes.issue_code(email)
    second, _ = login_codes.issue_code(email)
    assert first != second
    assert login_codes.verify_code(email, first) is False
    assert login_codes.verify_code(email, second) is True


def test_resend_is_refused_inside_the_cooldown():
    email = _address()
    login_codes.issue_code(email)
    with pytest.raises(login_codes.CodeRequestRefused) as excinfo:
        login_codes.issue_code(email)
    assert excinfo.value.retry_after > 0


def test_purge_expired_removes_only_dead_codes(monkeypatch):
    alive, dead = _address(), _address()
    login_codes.issue_code(alive)
    login_codes.issue_code(dead)
    login_codes._conn().execute(
        "UPDATE login_codes SET expires_at = 1 WHERE email = ?", (dead,)
    )
    login_codes._conn().commit()

    login_codes.purge_expired()
    assert login_codes.debug_state(alive) is not None
    assert login_codes.debug_state(dead) is None


# ── Over HTTP ────────────────────────────────────────────────────────

def test_requesting_a_code_then_signing_in_creates_the_account(client, sent):
    client.cookies.clear()
    email = _address()

    resp = client.post("/api/v1/auth/code/request", json={"email": email})
    assert resp.status_code == 200, resp.text
    assert resp.json()["sent"] is True
    assert resp.json()["expires_in_minutes"] >= 1
    assert len(sent) == 1 and sent[0][0] == email

    code = sent[0][1]
    resp = client.post("/api/v1/auth/code/verify", json={"email": email, "code": code})
    assert resp.status_code == 201, resp.text  # 201: the account is new
    assert resp.json()["username"] == email
    assert client.cookies.get("user_id"), "no session cookie after verifying"
    client.cookies.clear()


def test_signing_in_again_reuses_the_same_account(client, sent, monkeypatch):
    monkeypatch.setattr(
        login_codes.get_settings(), "login_code_resend_seconds", 0, raising=False
    )
    client.cookies.clear()
    email = _address()

    client.post("/api/v1/auth/code/request", json={"email": email})
    first = client.post(
        "/api/v1/auth/code/verify", json={"email": email, "code": sent[-1][1]}
    )
    assert first.status_code == 201
    client.cookies.clear()

    client.post("/api/v1/auth/code/request", json={"email": email})
    again = client.post(
        "/api/v1/auth/code/verify", json={"email": email, "code": sent[-1][1]}
    )
    assert again.status_code == 200, "a second sign-in made a second account"
    assert again.json()["user_id"] == first.json()["user_id"]
    client.cookies.clear()


def test_the_address_is_matched_case_insensitively_over_http(client, sent):
    client.cookies.clear()
    email = _address()
    client.post("/api/v1/auth/code/request", json={"email": email.upper()})
    resp = client.post(
        "/api/v1/auth/code/verify", json={"email": email, "code": sent[-1][1]}
    )
    assert resp.status_code in (200, 201), resp.text
    assert resp.json()["username"] == email.lower()
    client.cookies.clear()


def test_a_wrong_code_gives_401_and_no_session(client, sent):
    client.cookies.clear()
    email = _address()
    client.post("/api/v1/auth/code/request", json={"email": email})

    resp = client.post("/api/v1/auth/code/verify", json={"email": email, "code": "000000"})
    assert resp.status_code == 401
    assert not client.cookies.get("user_id"), "session issued on a wrong code"


def test_the_request_endpoint_does_not_reveal_who_has_an_account(client, sent):
    """Known and unknown addresses must be indistinguishable in the reply."""
    client.cookies.clear()
    known = _address()
    client.post("/api/v1/auth/code/request", json={"email": known})
    client.post("/api/v1/auth/code/verify", json={"email": known, "code": sent[-1][1]})
    client.cookies.clear()

    unknown = client.post("/api/v1/auth/code/request", json={"email": _address()})
    # The known address is inside its cooldown, so compare shapes not codes.
    assert unknown.status_code == 200
    assert set(unknown.json()) == {"sent", "delivered", "expires_in_minutes", "resend_in_seconds"}


@pytest.mark.parametrize("bad", ["", "nope", "nope@", "@example.test", "a b@c.test"])
def test_malformed_addresses_are_rejected_before_anything_is_sent(client, sent, bad):
    resp = client.post("/api/v1/auth/code/request", json={"email": bad})
    assert resp.status_code == 422
    assert sent == [], "a code was issued for an address that cannot exist"


def test_resend_inside_the_cooldown_returns_429_with_retry_after(client, sent):
    client.cookies.clear()
    email = _address()
    client.post("/api/v1/auth/code/request", json={"email": email})
    resp = client.post("/api/v1/auth/code/request", json={"email": email})
    assert resp.status_code == 429
    assert 0 < int(resp.headers["Retry-After"]) <= 600


def test_a_failed_send_does_not_cost_the_cooldown(client, monkeypatch):
    """A provider outage must not leave someone waiting with no code."""
    import app.api.routes.auth as auth_routes

    async def _boom(*_args, **_kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(auth_routes, "send_login_code", _boom)
    email = _address()

    resp = client.post("/api/v1/auth/code/request", json={"email": email})
    assert resp.status_code == 502
    assert login_codes.debug_state(email) is None, "a code outlived a failed send"

    # And the next attempt is allowed immediately, not after the cooldown.
    box: list = []

    async def _ok(to_email, code, ttl):
        box.append(code)

    monkeypatch.setattr(auth_routes, "send_login_code", _ok)
    assert client.post("/api/v1/auth/code/request", json={"email": email}).status_code == 200


def test_the_daily_cap_stops_the_endpoint_being_used_as_a_mail_cannon(client, sent, monkeypatch):
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "login_code_daily_cap", 0, raising=False)
    resp = client.post("/api/v1/auth/code/request", json={"email": _address()})
    assert resp.status_code == 429
    assert "today" in resp.json()["detail"].lower()


def test_an_account_made_by_code_cannot_be_entered_with_a_blank_password(client, sent):
    """No password was ever set, so the password door stays shut."""
    client.cookies.clear()
    email = _address()
    client.post("/api/v1/auth/code/request", json={"email": email})
    client.post("/api/v1/auth/code/verify", json={"email": email, "code": sent[-1][1]})
    client.cookies.clear()

    for attempt in (" ", "password", "correct-horse-battery", ":"):
        resp = client.post("/api/v1/auth/login", json={"username": email, "password": attempt})
        assert resp.status_code == 401, f"blank-password account opened with {attempt!r}"
        assert not client.cookies.get("user_id")
    client.cookies.clear()


# ── With no mail provider configured ─────────────────────────────────

async def _send(email: str, code: str, ttl: int) -> None:
    from app.core.mailer import send_login_code

    await send_login_code(email, code, ttl)


def test_development_logs_the_code_instead_of_sending(caplog):
    """The flow has to be usable before a provider exists."""
    import asyncio

    from app.config import get_settings

    assert get_settings().app_env == "development"  # set by conftest
    assert not get_settings().email_sending_configured

    asyncio.run(_send("ada@example.test", "424242", 10))
    # The one place a live code is written down, and only here.
    assert any("login_code_not_emailed_dev_only" in r.message for r in caplog.records)


def test_production_without_mail_refuses_rather_than_pretending(monkeypatch):
    """Logging a code the reader cannot see, while claiming mail is on its
    way, is worse than an honest failure."""
    import asyncio

    from app.config import get_settings
    from app.core.mailer import EmailNotConfigured

    settings = get_settings()
    monkeypatch.setattr(settings, "app_env", "production", raising=False)
    monkeypatch.setattr(settings, "smtp_host", "", raising=False)

    with pytest.raises(EmailNotConfigured):
        asyncio.run(_send("ada@example.test", "424242", 10))


def test_the_endpoint_answers_503_when_it_cannot_send(client, monkeypatch):
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "app_env", "production", raising=False)
    monkeypatch.setattr(settings, "smtp_host", "", raising=False)

    email = _address()
    resp = client.post("/api/v1/auth/code/request", json={"email": email})
    assert resp.status_code == 503, resp.text
    assert "password" in resp.json()["detail"].lower(), "no way forward offered"
    assert login_codes.debug_state(email) is None, "a code was kept for a mail never sent"


def test_the_reply_says_when_a_code_was_not_actually_delivered(client, sent):
    """Development with no provider: the client must not say 'check your inbox'."""
    resp = client.post("/api/v1/auth/code/request", json={"email": _address()})
    assert resp.status_code == 200
    assert resp.json()["delivered"] is False


def test_both_doors_open_the_same_account(client, sent, monkeypatch):
    """Password and code are two ways into one account, not two accounts."""
    monkeypatch.setattr(
        login_codes.get_settings(), "login_code_resend_seconds", 0, raising=False
    )
    client.cookies.clear()
    email = _address()
    password = "correct-horse-battery"

    registered = client.post(
        "/api/v1/auth/register", json={"username": email, "password": password}
    )
    assert registered.status_code == 201, registered.text
    client.cookies.clear()

    client.post("/api/v1/auth/code/request", json={"email": email})
    by_code = client.post(
        "/api/v1/auth/code/verify", json={"email": email, "code": sent[-1][1]}
    )
    assert by_code.status_code == 200, "signing in by code made a second account"
    assert by_code.json()["user_id"] == registered.json()["user_id"]
    client.cookies.clear()

    by_password = client.post(
        "/api/v1/auth/login", json={"username": email, "password": password}
    )
    assert by_password.status_code == 200
    assert by_password.json()["user_id"] == registered.json()["user_id"]
    client.cookies.clear()
