"""Centralized, typed application configuration.

All settings (API keys, model names, storage paths) live here in one place,
loaded from environment variables / the .env file via pydantic-settings.
This validates types at startup and fails loudly if a required secret
(GOOGLE_API_KEY) is missing, instead of crashing cryptically mid-run.
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, sourced from the environment and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated env vars instead of erroring
    )

    # --- Secrets (required: no default -> startup error if absent) ---
    # SecretStr prevents the key from being printed in logs / tracebacks.
    google_api_key: SecretStr

    # --- Model choices (Gemini covers both embeddings and chat) ---
    embedding_model: str = "models/gemini-embedding-001"
    chat_model: str = "gemini-2.0-flash"

    # --- Vector store location ---
    chroma_dir: str = "data/chroma"
    collection_name: str = "paperwhisperer"


# A single shared instance imported across the app.
settings = Settings()
