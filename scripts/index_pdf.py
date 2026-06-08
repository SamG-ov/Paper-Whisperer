"""Index a PDF into the vector store: load -> chunk -> embed -> persist.

This runs the entire ingestion pipeline and calls the Gemini embedding API.

Usage:
    python scripts/index_pdf.py data/sample.pdf
"""

import sys

from paperwhisperer.ingestion.chunker import split_documents
from paperwhisperer.ingestion.loader import load_pdf
from paperwhisperer.vector_store import index_documents

sys.stdout.reconfigure(encoding="utf-8")  # UTF-8 for Windows consoles


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/index_pdf.py <path-to-pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    print(f"[1/3] Loading {pdf_path} ...")
    pages = load_pdf(pdf_path)

    print(f"[2/3] Chunking {len(pages)} pages ...")
    chunks = split_documents(pages)

    print(f"[3/3] Embedding + storing {len(chunks)} chunks (calls Gemini) ...")
    store = index_documents(chunks)
    print(f"      Stored. Collection now holds {store._collection.count()} vectors.")

    # --- Smoke test: prove the vectors are searchable (real retrieval is M4) ---
    query = "What is multi-head attention?"
    print(f"\nSmoke-test query: {query!r}")
    results = store.similarity_search(query, k=2)
    for i, doc in enumerate(results, 1):
        page = doc.metadata.get("page_label", "?")
        snippet = doc.page_content[:160].replace("\n", " ")
        print(f"  result {i} (page {page}): {snippet}...")


if __name__ == "__main__":
    main()
