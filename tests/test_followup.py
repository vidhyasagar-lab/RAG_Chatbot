"""At most one extra search, when the first round visibly misses part of the question.

A cheap check sees the question and each retrieved chunk's source, section and
first 100 characters - not the full text - and names up to three follow-up
searches. Their results are fused with the first round. There is never a
third round, and a check that fails in any way leaves the first round alone.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import app.core.followup as followup
from app.config import get_settings


def _doc(text: str, section: str = "", content_type: str = "text") -> Document:
    metadata = {"source": "uploads/u1/report.docx", "page": "", "content_type": content_type}
    if section:
        metadata["section_header"] = section
    return Document(page_content=text, metadata=metadata)


class _Completions:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _check_replies(monkeypatch, reply) -> _Completions:
    completions = _Completions(reply)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(followup, "_get_client", lambda: client)
    return completions


FIRST = [_doc("Growth by market", section="3 Growth")]


# ── plan_followups ───────────────────────────────────────────────────

def test_missing_evidence_produces_queries(monkeypatch):
    _check_replies(monkeypatch, json.dumps({"missing": True, "queries": ["ranking table"]}))

    assert followup.plan_followups("q", FIRST) == ["ranking table"]


def test_at_most_three_string_queries_are_used(monkeypatch):
    _check_replies(monkeypatch, json.dumps(
        {"missing": True, "queries": ["a", 5, "b", "  ", "c", "d"], "why": "extra key"}))

    assert followup.plan_followups("q", FIRST) == ["a", "b", "c"]


def test_a_string_instead_of_a_list_is_not_split_into_letters(monkeypatch):
    """Iterating "ranking table" would search for "r", "a" and "n"."""
    _check_replies(monkeypatch, json.dumps({"missing": True, "queries": "ranking table"}))

    assert followup.plan_followups("q", FIRST) == []


def test_the_client_never_retries():
    """The SDK retries twice by default, and honours Retry-After on a 429 for
    up to two minutes - all before the reader sees a single source."""
    client = followup._build_client()

    assert client.max_retries == 0


def test_covered_means_no_followup(monkeypatch):
    _check_replies(monkeypatch, json.dumps({"missing": False, "queries": ["ignored"]}))

    assert followup.plan_followups("q", FIRST) == []


@pytest.mark.parametrize("reply", ["not json", "[]", RuntimeError("azure down")],
                         ids=["bad-json", "wrong-shape", "raises"])
def test_a_failing_check_never_raises(monkeypatch, reply):
    _check_replies(monkeypatch, reply)

    assert followup.plan_followups("q", FIRST) == []


def test_disabled_means_no_check_at_all(monkeypatch):
    completions = _check_replies(monkeypatch, json.dumps({"missing": True, "queries": ["x"]}))
    monkeypatch.setattr(get_settings(), "followup_retrieval_enabled", False)

    assert followup.plan_followups("q", FIRST) == []
    assert completions.calls == []


def test_nothing_retrieved_means_nothing_to_check(monkeypatch):
    completions = _check_replies(monkeypatch, json.dumps({"missing": True, "queries": ["x"]}))

    assert followup.plan_followups("q", []) == []
    assert completions.calls == []


def test_the_check_is_bounded(monkeypatch):
    completions = _check_replies(monkeypatch, json.dumps({"missing": False, "queries": []}))
    followup.plan_followups("q", FIRST)

    sent = completions.calls[0]
    assert sent["timeout"] == 8.0
    assert sent["max_completion_tokens"] == 400
    assert sent["response_format"] == {"type": "json_object"}


# ── coverage_digest ──────────────────────────────────────────────────

def test_the_digest_names_the_section_and_cuts_the_body():
    body = "x" * 300
    digest = followup.coverage_digest(
        [_doc(f"[Document: report.docx | Section: 9 Markets > Ranking]\n{body}",
              section="9 Markets > Ranking", content_type="table")])

    assert "report.docx" in digest
    assert "9 Markets > Ranking" in digest
    assert "[table]" in digest
    assert "x" * 100 in digest and "x" * 101 not in digest


# ── retrieve_with_followup ───────────────────────────────────────────

def _searches(monkeypatch, second=None):
    calls = []
    extra = _doc("Composite ranking table", section="9 Markets > Ranking")

    def search(query, k=None, user_id=""):
        calls.append(query)
        return list(FIRST) if query == "q" else [second or extra]

    monkeypatch.setattr(followup, "hybrid_search", search)
    return calls, extra


def test_a_followup_brings_in_what_was_missing(monkeypatch):
    calls, extra = _searches(monkeypatch)
    monkeypatch.setattr(followup, "plan_followups", lambda q, docs: ["ranking"])
    searched = []

    docs = followup.retrieve_with_followup("q", k=12, user_id="u",
                                           on_search=lambda: searched.append(1))

    assert extra in docs
    assert calls == ["q", "ranking"]
    assert searched == [1]


def test_no_queries_means_the_first_round_unchanged(monkeypatch):
    calls, _ = _searches(monkeypatch)
    monkeypatch.setattr(followup, "plan_followups", lambda q, docs: [])
    searched = []

    docs = followup.retrieve_with_followup("q", k=12, on_search=lambda: searched.append(1))

    assert docs == FIRST
    assert calls == ["q"]
    assert searched == []


def test_a_followup_cannot_push_out_round_one_top_half():
    """Equal-weight fusion keeps ranks 1..k/(n+1) of each list: with three
    follow-ups, round one would keep only its top three, dropping an answer
    it had at rank five whenever the check asked for more."""
    first = [_doc(f"first {i}") for i in range(12)]
    extras = [[_doc(f"extra {j}")] for j in range(3)]

    fused = followup.fuse_rounds(first, extras, k=12)

    assert fused[:6] == first[:6]


def test_followup_hits_come_straight_after_the_protected_half():
    """Next in line, so the token budget - which trims from the end - keeps them."""
    first = [_doc(f"first {i}") for i in range(12)]
    extras = [[_doc(f"extra {j}")] for j in range(3)]

    fused = followup.fuse_rounds(first, extras, k=12)

    assert {d.page_content for d in fused[6:9]} == {"extra 0", "extra 1", "extra 2"}
    assert len(fused) == 12
