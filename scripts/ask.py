"""Ask a question about the indexed PDF and get a grounded, cited answer.

Usage:
    python scripts/ask.py "How does the Transformer handle word order?"
"""

import sys

from paperwhisperer.rag import build_rag_chain

sys.stdout.reconfigure(encoding="utf-8")  # UTF-8 for Windows consoles


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/ask.py "your question"')
        sys.exit(1)
    question = sys.argv[1]

    chain = build_rag_chain()
    result = chain.invoke(question)

    print(f"Q: {question}\n")
    print(f"A: {result['answer']}\n")

    # List the unique source pages the retriever supplied.
    def _page_key(label: str) -> int:
        return int(label) if label.isdigit() else 9999

    pages = sorted(
        {d.metadata.get("page_label", "?") for d in result["context"]},
        key=_page_key,
    )
    print("Sources: page(s)", ", ".join(pages))


if __name__ == "__main__":
    main()
