"""The admin bootstrap command: the only way to create the first admin."""

from __future__ import annotations

import uuid

import pytest

PASSWORD = "correct-horse-battery"


def _name() -> str:
    return f"adm_{uuid.uuid4().hex[:10]}"


def _answers(monkeypatch, *replies):
    """Feed getpass prompts in order; fail loudly if prompted more than expected."""
    it = iter(replies)

    def fake_getpass(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(f"unexpected password prompt: {prompt!r}")

    monkeypatch.setattr("getpass.getpass", fake_getpass)


def test_creates_a_new_admin(monkeypatch):
    from app.core.user_store import authenticate_user
    from app.scripts.create_admin import main

    name = _name()
    _answers(monkeypatch, PASSWORD, PASSWORD)
    assert main([name]) == 0
    user = authenticate_user(name, PASSWORD)
    assert user is not None
    assert user["role"] == "admin"


def test_promotes_an_existing_user_without_touching_the_password(monkeypatch):
    from app.core.user_store import authenticate_user, register_user
    from app.scripts.create_admin import main

    name = _name()
    register_user(name, PASSWORD)
    _answers(monkeypatch)  # any prompt is a failure
    assert main([name]) == 0
    user = authenticate_user(name, PASSWORD)
    assert user is not None, "promotion changed or broke the password"
    assert user["role"] == "admin"


def test_mismatched_confirmation_creates_nothing(monkeypatch):
    from app.core.user_store import authenticate_user
    from app.scripts.create_admin import main

    name = _name()
    _answers(monkeypatch, PASSWORD, PASSWORD + "x")
    assert main([name]) == 1
    assert authenticate_user(name, PASSWORD) is None


def test_short_password_creates_nothing(monkeypatch, capsys):
    from app.core.user_store import authenticate_user
    from app.scripts.create_admin import main

    name = _name()
    _answers(monkeypatch, "short", "short")
    assert main([name]) == 1
    assert "at least 8 characters" in capsys.readouterr().err
    assert authenticate_user(name, "short") is None


def test_bootstrapped_admin_can_reach_the_admin_api(client, monkeypatch):
    """The reason the command exists: /api/v1/admin/* was unreachable."""
    from app.scripts.create_admin import main

    name = _name()
    _answers(monkeypatch, PASSWORD, PASSWORD)
    assert main([name]) == 0

    client.cookies.clear()
    try:
        login = client.post("/api/v1/auth/login", json={"username": name, "password": PASSWORD})
        assert login.status_code == 200
        assert login.json()["role"] == "admin"
        assert client.get("/api/v1/admin/stats").status_code == 200
    finally:
        client.cookies.clear()
