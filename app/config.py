"""Application configuration via environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings

#: Shipped placeholder. Startup refuses to run with this outside development.
DEFAULT_SECRET_KEY = "change-me-to-a-random-secret"


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
    # Refused at startup outside development — see app/main.py lifespan.
    secret_key: str = DEFAULT_SECRET_KEY  # HMAC signing for session cookies

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
    eval_quality_threshold: float = 0.5  # minimum faithfulness to show answer
    eval_max_retries: int = 1  # how many times to regenerate on low quality

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
