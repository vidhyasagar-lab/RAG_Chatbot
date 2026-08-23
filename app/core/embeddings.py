"""Azure OpenAI Embedding wrapper."""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import AzureOpenAIEmbeddings

from app.config import get_settings


@lru_cache
def get_embeddings() -> AzureOpenAIEmbeddings:
    """Return a cached Azure OpenAI Embeddings instance."""
    settings = get_settings()
    return AzureOpenAIEmbeddings(
        azure_deployment=settings.azure_openai_embedding_model,
        azure_endpoint=settings.effective_embedding_endpoint,
        api_key=settings.effective_embedding_api_key,
        api_version=settings.effective_embedding_api_version,
    )
