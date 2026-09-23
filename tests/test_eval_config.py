"""The evaluator can be pointed at its own deployment, and ships telemetry-free.

Measured on 2026-09-23: gpt-4.1-mini is 1.8x SLOWER than gpt-5.2 on the
faithfulness metric, so these settings ship unset and gpt-5.2 stays the
evaluator. The seam exists so that choice can be revisited by env var
rather than by code change.
"""

from __future__ import annotations

import pytest

from app.config import Settings


def _settings(**overrides) -> Settings:
    """A Settings built from explicit values, ignoring the developer's .env."""
    base = dict(
        azure_openai_api_key="main-key",
        azure_openai_endpoint="https://main.openai.azure.com",
        azure_openai_model="gpt-5.2",
        azure_openai_api_version="2025-01-01-preview",
        secret_key="test-secret-not-the-default-value",
        _env_file=None,
    )
    base.update(overrides)
    return Settings(**base)


def test_eval_model_falls_back_to_the_main_chat_model():
    assert _settings().effective_eval_model == "gpt-5.2"


def test_eval_model_override_is_used_when_set():
    assert _settings(eval_model="gpt-4.1-mini").effective_eval_model == "gpt-4.1-mini"


def test_eval_endpoint_falls_back_to_the_main_endpoint():
    assert _settings().effective_eval_endpoint == "https://main.openai.azure.com"


def test_eval_endpoint_can_point_at_a_different_resource():
    """gpt-4.1-mini lives on ams-chatgpt, not the main resource."""
    s = _settings(
        eval_endpoint="https://ams-chatgpt.cognitiveservices.azure.com",
        eval_api_key="other-key",
    )
    assert s.effective_eval_endpoint == "https://ams-chatgpt.cognitiveservices.azure.com"
    assert s.effective_eval_api_key == "other-key"
    # Overriding the endpoint must not drag the model along with it.
    assert s.effective_eval_model == "gpt-5.2"


def test_eval_api_key_falls_back_to_the_main_key():
    assert _settings().effective_eval_api_key == "main-key"


def test_eval_api_version_falls_back_to_the_main_version():
    assert _settings().effective_eval_api_version == "2025-01-01-preview"


@pytest.mark.parametrize(
    "field, expected",
    [("eval_timeout_seconds", 120), ("eval_max_answer_chars", 6000)],
)
def test_gate_bounds_have_defaults(field, expected):
    """The gate is bounded in both time and input size."""
    assert getattr(_settings(), field) == expected


def test_importing_the_evaluator_disables_ragas_telemetry():
    """ragas phones home to t.explodinggradients.com and retries on DNS failure.

    Asserted against os.environ rather than a ragas internal because the
    variable is read by ragas at ITS import, which happens lazily inside the
    evaluator's functions - long after this module is imported.
    """
    import os

    import app.core.evaluator  # noqa: F401  (import is the behaviour under test)

    assert os.environ.get("RAGAS_DO_NOT_TRACK") == "true"
