"""Create the first admin, or promote an existing user to admin.

Registration always assigns the ``user`` role and the admin API requires an
existing admin, so this command is the only way in.

    uv run python -m app.scripts.create_admin <username>                         # local
    docker compose exec rag-chatbot python -m app.scripts.create_admin <username>  # server
"""

from __future__ import annotations

import argparse
import getpass
import sys

from app.core.user_store import admin_create_user, get_user_by_username, set_user_role


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.scripts.create_admin", description=__doc__.splitlines()[0])
    parser.add_argument("username")
    username = parser.parse_args(argv).username.strip()

    existing = get_user_by_username(username)
    if existing:
        if existing["role"] == "admin":
            print(f"{username} is already an admin.")
        else:
            set_user_role(existing["user_id"], "admin")
            print(f"Promoted {username} to admin.")
        return 0

    password = getpass.getpass(f"Password for new admin {username}: ")
    if getpass.getpass("Confirm password: ") != password:
        print("Passwords do not match. Nothing was created.", file=sys.stderr)
        return 1
    try:
        admin_create_user(username, password, role="admin")
    except ValueError as e:
        print(f"{e}. Nothing was created.", file=sys.stderr)
        return 1
    print(f"Created admin {username}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
