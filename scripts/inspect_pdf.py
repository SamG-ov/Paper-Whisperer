"""Manual smoke test: load a PDF and inspect what we extracted.

This is a developer tool, not part of the app. It lets us *see* the raw
output of our loader so we can judge extraction quality with our own eyes.

Usage:
    python scripts/inspect_pdf.py data/sample.pdf
"""

import sys

from paperwhisperer.ingestion.loader import load_pdf

# Windows consoles default to the legacy cp1252 encoding, which cannot print
# many Unicode characters found in real PDFs (e.g. "∗", "→", "—"). Force UTF-8
# so this dev tool can display extracted text faithfully. (The data itself is
# always correct Unicode; this only affects console output.)
sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/inspect_pdf.py <path-to-pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    docs = load_pdf(pdf_path)

    print(f"Loaded {len(docs)} page(s) from {pdf_path}\n")

    print("=== First page metadata ===")
    print(docs[0].metadata)

    print("\n=== First page text (first 600 chars) ===")
    print(docs[0].page_content[:600])

    total_chars = sum(len(d.page_content) for d in docs)
    print("\n=== Stats ===")
    print(f"Total pages:      {len(docs)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Avg chars/page:   {total_chars // len(docs):,}")


if __name__ == "__main__":
    main()
