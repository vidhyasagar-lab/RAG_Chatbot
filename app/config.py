"""Application configuration via environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings

#: Minimum accepted length for SECRET_KEY. `secrets.token_urlsafe(32)` yields 43.
MIN_SECRET_KEY_LENGTH = 16

#: Values that are *rejected*, never used as fallbacks. There is deliberately no
#: default SECRET_KEY: a signing key baked into source is a signing key every
#: reader of the repository knows, which makes session cookies forgeable. The
#: real value belongs in .env and nowhere else.
PLACEHOLDER_SECRETS = frozenset(
    {
        "change-me-to-a-random-secret",
        "changeme",
        "change-me",
        "secret",
        "supersecret",
        "your-secret-key",
        "test",
    }
)


class Settings(BaseSettings):
    """Centralised settings loaded from .env / environment."""

    # Azure OpenAI
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_model: str = "gpt-5.2"
    azure_openai_api_version: str = "2025-01-01-preview"
    azure_openai_embedding_model: str = "text-embedding-ada-002"

    # Embedding-specific overrides (fall back to main Azure OpenAI values)
    azure_openai_embedding_api_key: str = ""
    azure_openai_embedding_endpoint: str = ""
    azure_openai_embedding_api_version: str = ""

    @property
    def effective_embedding_api_key(self) -> str:
        return self.azure_openai_embedding_api_key or self.azure_openai_api_key

    @property
    def effective_embedding_endpoint(self) -> str:
        return self.azure_openai_embedding_endpoint or self.azure_openai_endpoint

    @property
    def effective_embedding_api_version(self) -> str:
        return self.azure_openai_embedding_api_version or self.azure_openai_api_version

    # Evaluator-specific overrides (fall back to main Azure OpenAI values).
    # Measured 2026-09-23 on the faithfulness metric: gpt-5.2 mean 24.0s,
    # gpt-4.1-mini mean 43.8s. The smaller model wins claim decomposition and
    # loses verification badly, so these ship unset and gpt-5.2 stays the
    # evaluator. The seam is here so the choice can change by env var.
    eval_model: str = ""
    eval_endpoint: str = ""
    eval_api_key: str = ""
    eval_api_version: str = ""

    @property
    def effective_eval_model(self) -> str:
        return self.eval_model or self.azure_openai_model

    @property
    def effective_eval_endpoint(self) -> str:
        return self.eval_endpoint or self.azure_openai_endpoint

    @property
    def effective_eval_api_key(self) -> str:
        return self.eval_api_key or self.azure_openai_api_key

    @property
    def effective_eval_api_version(self) -> str:
        return self.eval_api_version or self.azure_openai_api_version

    # App
    app_env: str = "production"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    api_key: str = ""  # Optional: set to require X-API-Key header on all requests
    # Per-IP request budget. Applies to everything except /api/v1/health
    # and /static/. Set rate_limit_enabled=false to disable entirely.
    rate_limit: str = "60/minute"
    rate_limit_enabled: bool = True
    # HMAC signing for session cookies. Required — must come from .env, with no
    # in-source default. Validated below.
    secret_key: str

    @field_validator("secret_key")
    @classmethod
    def _reject_weak_secret_key(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError(
                "SECRET_KEY is empty. Set it in .env — generate one with:\n"
                '  python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )
        if stripped.lower() in PLACEHOLDER_SECRETS:
            raise ValueError(
                f"SECRET_KEY is still the placeholder {stripped!r}. Session cookies "
                "signed with a publicly known key are trivially forgeable. Set a real "
                "value in .env — generate one with:\n"
                '  python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )
        if len(stripped) < MIN_SECRET_KEY_LENGTH:
            raise ValueError(
                f"SECRET_KEY is {len(stripped)} characters; at least "
                f"{MIN_SECRET_KEY_LENGTH} are required. Generate one with:\n"
                '  python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )
        return v

    # RAG – Hybrid Chunking (token-based)
    parent_chunk_tokens: int = 512
    parent_overlap_tokens: int = 50
    child_chunk_tokens: int = 128
    child_overlap_tokens: int = 16
    semantic_threshold: int = 85  # percentile for semantic boundary detection

    # RAG – Hybrid Retrieval
    top_k_results: int = 5
    dense_weight: float = 0.5
    sparse_weight: float = 0.5

    # LLM
    max_tokens: int = 2048
    temperature: float = 0.3

    # Vision (multimodal)
    vision_detail: str = "high"  # low | high | auto
    vision_max_tokens: int = 1024
    # How many image-description calls to run concurrently during document
    # preprocessing. These are network-bound, so raising this shortens ingest
    # almost linearly — bounded to stay under Azure per-deployment rate limits.
    vision_max_concurrency: int = 6

    # Eval-gated answers
    eval_gating_enabled: bool = True
    eval_quality_threshold: float = 0.5  # minimum faithfulness to keep an answer
    eval_max_retries: int = 1  # how many times to regenerate on low quality
    # The gate runs after the answer has streamed, so it no longer blocks the
    # reader — but it must still be bounded. A gate that hangs would hold the
    # SSE connection open forever; one fed an enormous answer costs time
    # proportional to the claim count it decomposes into.
    eval_timeout_seconds: int = 120
    eval_max_answer_chars: int = 6000

    # Langfuse Observability
    langfuse_enabled: bool = True
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def validated_vision_detail(self) -> str:
        val = self.vision_detail.lower()
        if val not in ("low", "high", "auto"):
            return "high"
        return val

    # CORS
    cors_origins: str = "http://localhost:3000,http://localhost:8000"

    # Paths
    vectorstore_dir: str = "data/vectorstore"
    # Relational stores (users, chats, evals) live here — deliberately beside
    # the vector index rather than inside it, so the two can be backed up,
    # cleared, or rebuilt independently.
    data_dir: str = "data"
    upload_dir: str = "uploads"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()  # type: ignore[call-arg]
