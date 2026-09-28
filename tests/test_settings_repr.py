"""Printing the settings never prints a secret.

pydantic's repr shows every field. A failing test that touched the settings
object printed the embedding key and the Langfuse secret key into the test
output, and from there into a session transcript.
"""

from __future__ import annotations

from app.config import Settings

SECRETS = {
    "azure_openai_api_key": "AZURE-SECRET-VALUE-1",
    "azure_openai_embedding_api_key": "EMBED-SECRET-VALUE-2",
    "eval_api_key": "EVAL-SECRET-VALUE-3",
    "api_key": "API-SECRET-VALUE-4",
    "secret_key": "SIGNING-SECRET-VALUE-5-long-enough-to-pass-validation",
    "langfuse_public_key": "pk-lf-PUBLIC-VALUE-6",
    "langfuse_secret_key": "sk-lf-SECRET-VALUE-7",
}


def test_printing_settings_never_shows_a_secret():
    settings = Settings(azure_openai_endpoint="https://example.test", **SECRETS)
    printed = repr(settings) + str(settings)

    leaked = [name for name, value in SECRETS.items() if value in printed]
    assert leaked == []


def test_the_secrets_are_still_readable_as_attributes():
    settings = Settings(azure_openai_endpoint="https://example.test", **SECRETS)

    assert settings.langfuse_secret_key == SECRETS["langfuse_secret_key"]
