"""LLM response caching.

A global SQLite-backed cache for LLM calls: identical prompts return the
stored response instantly, with no API call. This cuts cost/latency for
repeated questions and eval re-runs, and relieves free-tier rate limits.
"""

from __future__ import annotations

from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

from paperwhisperer.config import settings

_initialized = False


def setup_caching() -> None:
    """Install the global LLM cache once (idempotent)."""
    global _initialized
    if _initialized:
        return
    set_llm_cache(SQLiteCache(database_path=settings.llm_cache_path))
    _initialized = True
