"""Faithfulness is a property of the answer, so the cache key must include it.

Without the answer in the key, asking the same question twice against the
same retrieved chunks returns the FIRST answer's faithfulness score for the
SECOND answer. With temperature > 0 those answers differ, so the gate would
admit ungrounded text or reject sound text on a stale verdict.
"""

from __future__ import annotations

from app.core.eval_store import get_eval_cache, save_eval_cache

QUESTION = "what were the Q3 revenue drivers?"
CONTEXTS = ["Revenue rose 12% in Q3.", "Enterprise seats grew 4%."]

GROUNDED = "Revenue rose 12 percent, driven by 4 percent seat growth."
UNGROUNDED = "Revenue tripled because of the new pricing model."


def test_scores_are_returned_for_the_answer_they_scored():
    save_eval_cache(QUESTION, CONTEXTS, {"faithfulness": 0.91}, answer=GROUNDED)

    cached = get_eval_cache(QUESTION, CONTEXTS, answer=GROUNDED)

    assert cached is not None
    assert cached["faithfulness"] == 0.91


def test_a_different_answer_does_not_inherit_the_cached_score():
    """The regression this task exists for."""
    save_eval_cache(QUESTION, CONTEXTS, {"faithfulness": 0.91}, answer=GROUNDED)

    cached = get_eval_cache(QUESTION, CONTEXTS, answer=UNGROUNDED)

    assert cached is None, (
        "an ungrounded answer was handed the grounded answer's faithfulness score"
    )


def test_the_same_answer_to_a_different_question_is_a_miss():
    save_eval_cache(QUESTION, CONTEXTS, {"faithfulness": 0.91}, answer=GROUNDED)

    assert get_eval_cache("something else entirely", CONTEXTS, answer=GROUNDED) is None


def test_answer_independent_scores_can_still_be_cached_without_an_answer():
    """context_precision does not depend on the answer, so it keys on ''."""
    save_eval_cache(QUESTION, CONTEXTS, {"context_precision": 0.75})

    cached = get_eval_cache(QUESTION, CONTEXTS)

    assert cached is not None
    assert cached["context_precision"] == 0.75
