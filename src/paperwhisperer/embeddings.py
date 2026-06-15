"""Embeddings factory (the embeddings side of the provider-agnostic seam).

Wraps the Gemini embedder in a file-backed cache so identical text is never
re-embedded: re-indexing unchanged chunks or repeating a query skips the API
call entirely. Swapping vendors later means editing only this file.
"""

from __future__ import annotations

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from paperwhisperer.config import settings


def get_embeddings() -> Embeddings:
    """Return the configured embedding model, wrapped in a persistent cache.

    Uses Google Gemini's gemini-embedding-001, which maps text into a
    high-dimensional semantic vector space.
    """
    underlying = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key.get_secret_value(),
        # gRPC bundles its own TLS that truststore doesn't patch; REST uses
        # httpx + the OS trust store.
        transport="rest",
    )

    cache_store = LocalFileStore(settings.embedding_cache_dir)
    return CacheBackedEmbeddings.from_bytes_store(
        underlying,
        cache_store,
        # Namespace keys by model so different models never collide. Sanitize
        # the "/" in the model id so it's a safe cache key.
        namespace=settings.embedding_model.replace("/", "_"),
        query_embedding_cache=True,  # also cache query embeddings
        key_encoder="sha256",  # collision-resistant cache keys
    )
