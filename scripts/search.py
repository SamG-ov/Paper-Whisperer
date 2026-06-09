"""Query the indexed PDF and inspect retrieval quality.

Shows relevance scores (your #1 debugging tool) and compares plain
similarity search against MMR (diverse) retrieval for the same query.

Usage:
    python scripts/search.py "How does the Transformer handle word order?"
"""

import sys

from paperwhisperer.retriever import get_retriever
from paperwhisperer.vector_store import get_vector_store

sys.stdout.reconfigure(encoding="utf-8")  # UTF-8 for Windows consoles


def _show(doc, score=None) -> None:
    page = doc.metadata.get("page_label", "?")
    snippet = doc.page_content[:150].replace("\n", " ")
    prefix = f"[score={score:.3f}] " if score is not None else ""
    print(f"  {prefix}(page {page}) {snippet}...")


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/search.py "your question"')
        sys.exit(1)
    query = sys.argv[1]
    print(f"Query: {query!r}\n")

    store = get_vector_store()

    # 1) Similarity search WITH relevance scores (0..1) -> debugging view.
    print("=== similarity (with relevance scores) ===")
    for doc, score in store.similarity_search_with_relevance_scores(query, k=4):
        _show(doc, score)

    # 2) The retriever abstraction with MMR for diverse results.
    print("\n=== mmr via retriever (diverse; fetch_k=20) ===")
    docs = get_retriever(k=4, search_type="mmr", fetch_k=20).invoke(query)
    for doc in docs:
        _show(doc)


if __name__ == "__main__":
    main()
