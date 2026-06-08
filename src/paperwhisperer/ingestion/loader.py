"""Load PDF files into LangChain Document objects.

This is the first stage of the ingestion pipeline. It is deliberately thin:
its only job is to read a PDF from disk and hand back a list of Documents
(one per page), preserving source/page metadata for later citation.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str | Path) -> list[Document]:
    """Load a PDF from disk into a list of Documents (one per page).

    Args:
        file_path: Path to a ``.pdf`` file.

    Returns:
        A list of ``Document`` objects. Each Document has:
          - ``page_content``: the extracted text of one page
          - ``metadata``: ``{"source": <path>, "page": <0-based page index>, ...}``

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if the path is not a ``.pdf`` file.
    """
    path = Path(file_path)

    # Fail fast with a clear error instead of letting a cryptic one surface
    # from deep inside the PDF library.
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix!r}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()
    return documents
