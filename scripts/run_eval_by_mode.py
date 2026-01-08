#!/usr/bin/env python3
"""Run evaluation across embedding modes: mock, tfidf, openai.

This script sets `EMBEDDING_MODE` for each run, imports the evaluation
runner and executes it. Results are printed and saved to
`logs/eval_by_mode_<mode>.json` for each mode.

Note: The vectorizer for TF-IDF must be built before running tfidf mode
(if not present, the retriever will fit it when building an in-memory index).
"""
from __future__ import annotations
import os
import json
from pathlib import Path

MODES = ["mock", "tfidf", "openai"]
OUT_DIR = Path("logs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for mode in MODES:
    print(f"\n=== Running evaluation with EMBEDDING_MODE={mode} ===")
    os.environ["EMBEDDING_MODE"] = mode
    # Importing here ensures the module picks up EMBEDDING_MODE when it needs to
    from src.evaluation.run_eval import evaluate

    results = evaluate()
    # results is a dict mapping top_k -> (summary, per_query_results)
    out_file = OUT_DIR / f"eval_by_mode_{mode}.json"
    serializable = {}
    for k, (summary, per_q) in results.items():
        serializable[str(k)] = {"summary": summary, "per_query": per_q}
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2)
    print(f"Saved results to {out_file}")

print("\nAll modes complete.")
