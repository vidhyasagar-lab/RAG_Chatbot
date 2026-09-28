"""Retrieve more, but never let the context grow without a bound.

top_k=5 cut off the table that answers three of the review questions; it first
appears at rank 6-15. Twelve results with parent expansion can reach ~6,000
tokens, so a budget trims from the lowest-ranked end.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

import app.core.rag_engine as engine
from app.config import Settings, get_settings
from app.core.document_loader import _token_len

ROOT = Path(__file__).resolve().parents[1]


def _doc(words: int, tag: str) -> Document:
    return Document(page_content=f"{tag} " + "word " * words,
                    metadata={"source": f"{tag}.pdf", "page": 1})


def test_twelve_results_by_default():
    assert Settings.model_fields["top_k_results"].default == 12


def test_the_example_env_does_not_pin_the_old_value():
    """Deployments copy .env.example; a stale 5 there silently wins."""
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    assert "TOP_K_RESULTS=12" in lines


def test_the_budget_defaults_to_six_thousand_tokens():
    assert Settings.model_fields["max_context_tokens"].default == 6000


def test_the_budget_keeps_a_rank_order_prefix():
    docs = [_doc(100, "a"), _doc(100, "b"), _doc(100, "c")]
    budget = _token_len(docs[0].page_content) + _token_len(docs[1].page_content)

    kept = engine._apply_budget(docs, budget)

    assert [d.metadata["source"] for d in kept] == ["a.pdf", "b.pdf"]


def test_an_oversized_first_result_is_still_kept():
    """An empty context is worse than an oversized one."""
    kept = engine._apply_budget([_doc(500, "big"), _doc(5, "small")], 50)

    assert [d.metadata["source"] for d in kept] == ["big.pdf"]


def test_build_context_applies_the_budget(monkeypatch):
    docs = [_doc(100, "first"), _doc(100, "second")]
    monkeypatch.setattr(engine, "hybrid_search", lambda *a, **kw: docs)
    monkeypatch.setattr(get_settings(), "max_context_tokens", 120)

    text, sources, _ = engine._build_context("q", user_id="u")

    assert "first" in text and "second" not in text
    assert [s["source"] for s in sources] == ["first.pdf"]
