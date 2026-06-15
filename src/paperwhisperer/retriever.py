"""Retrieval: turn the persisted vector store into a query-able retriever.

A Retriever is LangChain's standard interface for "given a query, return
relevant Documents" (via .invoke). Building on it -- instead of calling
similarity_search directly -- lets us swap strategies (similarity / MMR /
score-threshold) and plug straight into generation chains in Milestone 5.
"""

from __future__ import annotations

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever

from paperwhisperer.config import settings
from paperwhisperer.vector_store import get_vector_store

try:  # Chroma type is only needed for the optional `store` arg's annotation
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover
    Chroma = None  # type: ignore


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


def _all_documents(store) -> list[Document]:
    """Pull every stored chunk back out of Chroma (to build a BM25 index)."""
    data = store.get()  # includes 'documents' and 'metadatas' by default
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]


def get_hybrid_retriever(k: int | None = None, store=None) -> BaseRetriever:
    """Combine semantic (vector) and keyword (BM25) retrieval via RRF.

    Semantic search matches meaning; BM25 matches exact tokens (numbers,
    names, jargon). EnsembleRetriever fuses both rankings with Reciprocal
    Rank Fusion, so a chunk ranked highly by *either* method surfaces.

    Args:
        k: results per retriever (defaults to settings.top_k).
        store: an existing Chroma store to reuse (e.g. a UI-cached client).
    """
    k = k or settings.top_k
    store = store or get_vector_store()

    vector_retriever = store.as_retriever(search_kwargs={"k": k})

    bm25 = BM25Retriever.from_documents(_all_documents(store))
    bm25.k = k

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25],
        weights=[0.5, 0.5],
    )


def retrieve(query: str, **kwargs: object) -> list[Document]:
    """Convenience helper: retrieve relevant chunks for a query string."""
    return get_retriever(**kwargs).invoke(query)
