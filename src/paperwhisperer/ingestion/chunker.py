"""Split Documents into smaller, retrieval-friendly chunks.

Second stage of the ingestion pipeline. Takes the page-level Documents from
the loader and breaks them into overlapping chunks that are small enough to
embed precisely but large enough to retain meaning. Metadata (source, page)
is carried onto every chunk so we can still cite sources after splitting.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Defaults chosen as a sensible starting point for prose/research papers.
# These are meant to be *tuned by measurement* later, not treated as sacred.
# ~1000 chars ≈ ~250 tokens, well under any embedding token limit.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split Documents into overlapping chunks.

    Args:
        documents: Documents to split (e.g. the output of ``load_pdf``).
        chunk_size: Target maximum size of each chunk, in characters.
        chunk_overlap: Number of characters shared between consecutive chunks.

    Returns:
        A list of chunk Documents. Each retains the parent's metadata plus a
        ``start_index`` marking where the chunk begins in its source page.

    Raises:
        ValueError: if chunk_overlap >= chunk_size (would loop / make no sense).
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size})."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,  # record each chunk's offset within its source page
    )
    return splitter.split_documents(documents)
