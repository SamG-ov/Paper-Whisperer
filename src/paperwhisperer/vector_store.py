"""Vector store: persist chunk embeddings in ChromaDB and query them.

ChromaDB runs in-process and persists to disk under settings.chroma_dir, so
the index survives between runs (no server to manage). We give each chunk a
deterministic ID so re-indexing the same PDF upserts instead of duplicating.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from paperwhisperer.config import settings
from paperwhisperer.embeddings import get_embeddings


def get_vector_store() -> Chroma:
    """Open (or create) the persistent Chroma collection."""
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
    )


def _chunk_id(doc: Document) -> str:
    """Build a stable ID from a chunk's origin so re-runs upsert cleanly."""
    meta = doc.metadata
    return f"{meta.get('source')}:{meta.get('page')}:{meta.get('start_index')}"


def index_documents(chunks: list[Document]) -> Chroma:
    """Embed and store chunks, returning the (now-populated) vector store.

    Embedding happens inside add_documents: Chroma calls our embeddings
    function on each chunk's text and stores the resulting vectors. Passing
    deterministic ids makes this idempotent (upsert, not append).
    """
    store = get_vector_store()
    ids = [_chunk_id(c) for c in chunks]
    store.add_documents(documents=chunks, ids=ids)
    return store
