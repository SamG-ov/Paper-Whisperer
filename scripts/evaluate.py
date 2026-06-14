"""Evaluate the RAG pipeline against a small labelled dataset.

Usage:
    python scripts/evaluate.py                 # uses eval/dataset.json
    python scripts/evaluate.py eval/dataset.json
"""

import json
import sys
from pathlib import Path
from statistics import mean

from paperwhisperer.evaluation import evaluate_dataset

sys.stdout.reconfigure(encoding="utf-8")  # UTF-8 for Windows consoles


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/dataset.json")
    dataset = json.loads(path.read_text(encoding="utf-8"))

    print(f"Evaluating {len(dataset)} questions (this calls the API)...\n")
    results = evaluate_dataset(dataset)

    # Per-question scorecard.
    print(f"{'faith':>5} {'rel':>4} {'corr':>4} {'sec':>6}  question")
    print("-" * 70)
    for r in results:
        print(
            f"{r['faithfulness']:>5} {r['relevancy']:>4} {r['correctness']:>4} "
            f"{r['latency_s']:>6}  {r['question'][:48]}"
        )

    # Aggregates.
    print("\n=== Averages ===")
    for metric in ("faithfulness", "relevancy", "correctness"):
        print(f"  {metric:<14} {mean(r[metric] for r in results):.2f} / 5")
    print(f"  {'avg latency':<14} {mean(r['latency_s'] for r in results):.2f} s")


if __name__ == "__main__":
    main()
