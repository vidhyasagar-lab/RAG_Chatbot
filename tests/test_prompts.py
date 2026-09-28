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


@PROMPTS
def test_no_example_comes_from_the_review_document(prompt):
    lowered = prompt.lower()
    assert [t for t in REVIEW_TERMS if t in lowered] == []
