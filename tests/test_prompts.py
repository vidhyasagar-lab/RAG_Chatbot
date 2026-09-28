"""Both prompts allow reasoning from named evidence, and neither teaches to the test.

The answer prompt said "do not speculate", so a question whose answer follows
from a retrieved table got "the context does not say". The retry prompt said
"Nothing inferred", so a gate rejection of a correctly reasoned answer could
only produce a worse one.

The worked examples must not come from the review document: an example built
from the test set teaches to the test and makes the re-test meaningless.
"""

from __future__ import annotations

import pytest

from app.core.rag_engine import REFINED_SYSTEM_PROMPT, SYSTEM_PROMPT

PROMPTS = pytest.mark.parametrize("prompt", [SYSTEM_PROMPT, REFINED_SYSTEM_PROMPT],
                                  ids=["answer", "retry"])

REVIEW_TERMS = ["attractiveness", "interest rate", "india", "offshore", "renewable"]


@PROMPTS
def test_the_context_slot_formats_cleanly(prompt):
    assert "CONTEXT-SENTINEL" in prompt.format(context="CONTEXT-SENTINEL")


@PROMPTS
def test_the_context_is_fenced_as_untrusted(prompt):
    assert "--- CONTEXT BEGINS ---" in prompt
    assert "--- CONTEXT ENDS ---" in prompt
    assert "untrusted" in prompt


@PROMPTS
def test_an_entailed_answer_must_name_its_evidence(prompt):
    assert "name the facts" in prompt


@PROMPTS
def test_inference_from_the_context_is_not_forbidden(prompt):
    lowered = prompt.lower()
    assert "nothing inferred" not in lowered
    assert "do not speculate" not in lowered


@PROMPTS
def test_outside_knowledge_is_still_forbidden(prompt):
    assert "outside knowledge" in prompt.lower()


def test_the_answer_prompt_does_not_invite_case_labels():
    """Bullets written as "Stated: ..." taught the model to open an answer with
    "Unsupported:" - it printed the label it had been given."""
    import re

    assert not re.search(r"^\s*-\s*(Stated|Entailed|Unsupported)\s*:", SYSTEM_PROMPT, re.M)
    assert "Never label" in SYSTEM_PROMPT


def test_a_judgment_question_is_answered_from_the_facts_to_compare():
    """"Which best balances X, Y and Z?" was refused because no single number
    combines them, though every fact needed to compare was retrieved."""
    assert "judgment" in SYSTEM_PROMPT
    assert "what the choice is based on" in SYSTEM_PROMPT


def test_a_total_is_summed_from_its_listed_parts():
    """"What share of total additions is X?" was refused because no total was
    printed, though every part of it was retrieved."""
    assert "add them up" in SYSTEM_PROMPT
    assert "assume" in SYSTEM_PROMPT.lower()
    # The worked example computes shares from parts: 180 + 120 + 100 = 400.
    assert "400" in SYSTEM_PROMPT and "45%" in SYSTEM_PROMPT


def test_an_explanation_question_is_answered_from_the_facts():
    """"Why might X overtake Y?" opened with "the context doesn't explain why"
    and never drew the conclusion its own bullets supported."""
    assert "why might" in SYSTEM_PROMPT.lower()
    assert "Bad:  The context does not explain" in SYSTEM_PROMPT


def test_an_explanation_is_offered_as_what_the_facts_suggest():
    """Reasons built from facts are the model's inference, not a stated cause;
    teaching it to assert them as fact invites a confident causal story."""
    assert "what the facts suggest" in SYSTEM_PROMPT
    assert "could outweigh" in SYSTEM_PROMPT


def test_the_retry_prompt_answers_explanation_questions_from_the_facts():
    """After a rejection the rewrite of "which risks could raise costs, and
    why?" opened with "the context does not explain why" - the hedge the
    answer prompt had already been cured of - and dropped two risks."""
    lowered = REFINED_SYSTEM_PROMPT.lower()
    assert "why might" in lowered
    assert "do not open by saying the context does not explain" in lowered


def test_the_retry_prompt_forbids_quoted_evidence_tags():
    """"Name the facts it rests on" came back as a "(stated risk: "...")" tag
    after every bullet - evidence quoted at the reader instead of used."""
    assert "plain words" in REFINED_SYSTEM_PROMPT
    assert '"(stated' in REFINED_SYSTEM_PROMPT


@PROMPTS
def test_no_example_comes_from_the_review_document(prompt):
    lowered = prompt.lower()
    assert [t for t in REVIEW_TERMS if t in lowered] == []
