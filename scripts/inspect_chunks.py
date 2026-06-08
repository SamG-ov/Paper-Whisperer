"""Manual smoke test: load a PDF, chunk it, and inspect the result.

Lets us *feel* the chunking tradeoffs: how many chunks we get, their size
distribution, the metadata carried onto each, and what the overlap actually
looks like between two consecutive chunks.

Usage:
    python scripts/inspect_chunks.py data/sample.pdf
"""

import sys

from paperwhisperer.ingestion.chunker import split_documents
from paperwhisperer.ingestion.loader import load_pdf

# Windows consoles default to cp1252; force UTF-8 so we can print real text.
sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/inspect_chunks.py <path-to-pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    pages = load_pdf(pdf_path)
    chunks = split_documents(pages)

    sizes = [len(c.page_content) for c in chunks]
    print(f"Pages: {len(pages)}  ->  Chunks: {len(chunks)}")
    print(f"Chunk size (chars): min={min(sizes)} max={max(sizes)} "
          f"avg={sum(sizes) // len(sizes)}\n")

    print("=== A sample chunk (chunk #3) ===")
    sample = chunks[3]
    print("metadata:", sample.metadata)
    print("content:")
    print(sample.page_content)

    print("\n=== Overlap demo: end of chunk #3 vs start of chunk #4 ===")
    print("...END of #3 :", repr(chunks[3].page_content[-120:]))
    print("START of #4..:", repr(chunks[4].page_content[:120]))


if __name__ == "__main__":
    main()
