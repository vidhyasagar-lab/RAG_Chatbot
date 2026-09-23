"""The stores survive concurrent requests.

Each store used to hold one sqlite3 connection shared by every thread. Writes
took a lock but reads did not, and a sqlite3 connection must not be used by two
threads at once: overlapping requests (the admin overview fires four together,
each resolving the signed-in user) failed with
``sqlite3.InterfaceError: bad parameter or other API misuse``. Seen live on
2026-09-23 while an evaluation run was writing results in the background.
"""

from __future__ import annotations

import threading
import uuid

ROUNDS = 150
READERS = 8


def _hammer(read, write) -> list[BaseException]:
    errors: list[BaseException] = []
    start = threading.Barrier(READERS + 1)

    def reader():
        start.wait()
        try:
            for _ in range(ROUNDS):
                read()
        except BaseException as e:  # noqa: BLE001 - the point is to catch anything
            errors.append(e)

    def writer():
        start.wait()
        try:
            for _ in range(ROUNDS // 3):
                write()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(READERS)] + [threading.Thread(target=writer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_user_lookups_from_many_threads_while_documents_are_written(client):
    from app.core.user_store import get_user, record_document, register_user

    user = register_user(f"thr_{uuid.uuid4().hex[:10]}", "correct-horse-battery")
    errors = _hammer(
        read=lambda: get_user(user["user_id"]),
        write=lambda: record_document(user["user_id"], f"{uuid.uuid4().hex}.pdf", 10, 1),
    )
    assert errors == [], f"{len(errors)} failures, first: {errors[0]!r}"


def test_chat_reads_from_many_threads_while_messages_are_written(client):
    from app.core.chat_store import add_message, create_session, get_session_messages

    session = create_session("thread-test-user", "t")
    errors = _hammer(
        read=lambda: get_session_messages(session["session_id"]),
        write=lambda: add_message(session["session_id"], "user", "hi"),
    )
    assert errors == [], f"{len(errors)} failures, first: {errors[0]!r}"


def test_eval_reads_from_many_threads_while_results_are_written(client):
    from app.core.eval_store import create_eval_run, get_eval_runs, save_eval_result

    run_id = create_eval_run(total_samples=1)
    errors = _hammer(
        read=get_eval_runs,
        write=lambda: save_eval_result(run_id, "q", "a", ["c"], "g", {"faithfulness": 1.0}),
    )
    assert errors == [], f"{len(errors)} failures, first: {errors[0]!r}"


def test_each_thread_gets_its_own_connection(client):
    from app.core import user_store

    seen: dict[str, int] = {}

    def grab(name: str):
        seen[name] = id(user_store._get_conn())

    threads = [threading.Thread(target=grab, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen["a"] != seen["b"]
    # ...and a thread reuses its own rather than opening one per call.
    assert user_store._get_conn() is user_store._get_conn()
