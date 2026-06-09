"""Retrieval: turn the persisted vector store into a query-able retriever.

A Retriever is LangChain's standard interface for "given a query, return
relevant Documents" (via .invoke). Building on it -- instead of calling
similarity_search directly -- lets us swap strategies (similarity / MMR /
score-threshold) and plug straight into generation chains in Milestone 5.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from paperwhisperer.config import settings
from paperwhisperer.vector_store import get_vector_store


def get_retriever(
    k: int | None = None,
    search_type: str = "similarity",
    **search_kwargs: object,
) -> VectorStoreRetriever:
    """Build a retriever over the vector store.

    Args:
        k: number of chunks to return (defaults to settings.top_k).
        search_type: "similarity", "mmr", or "similarity_score_threshold".
        **search_kwargs: extra options for the strategy, e.g. fetch_k and
            lambda_mult (mmr), score_threshold, or filter (metadata filter).

    Returns:
        A VectorStoreRetriever ready to .invoke(query).
    """
    store = get_vector_store()
    search_kwargs = {"k": k or settings.top_k, **search_kwargs}
    return store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


def retrieve(query: str, **kwargs: object) -> list[Document]:
    """Convenience helper: retrieve relevant chunks for a query string."""
    return get_retriever(**kwargs).invoke(query)
