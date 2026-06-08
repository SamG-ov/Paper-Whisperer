"""Embeddings factory.

The rest of the app asks this module for "an embeddings object" and never
names the vendor directly. Swapping Gemini for OpenAI/Ollama later means
changing only this file -- the dependency-inversion principle in practice.
"""

from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from paperwhisperer.config import settings


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return the configured embedding model.

    Uses Google Gemini's gemini-embedding-001, which maps text into a
    high-dimensional semantic vector space.
    """
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key.get_secret_value(),
        # Use REST instead of the default gRPC transport. gRPC bundles its own
        # TLS stack and on Windows/Anaconda often can't find the system root
        # CAs (CERTIFICATE_VERIFY_FAILED). REST goes through httpx + certifi,
        # which validates correctly.
        transport="rest",
    )
