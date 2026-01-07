"""Evaluation runner: run retriever against gold testset and compute metrics.

Usage: run this module to evaluate the FAISS retriever on the small gold set.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import List

from src.evaluation.schemas import RAGTestSet
from src.evaluation import metrics
from src.retrieval.faiss_retriever import (
    load_retrieval_assets,
    retrieve_top_k,
    rerank_by_cosine,
)
from src.evaluation.validation import validate_testset_ids, confirm_ids


TESTSET_PATH = Path("data") / "rag_eval_testset.json"


def _extract_doc_ids(results: List[dict]) -> List[str]:
    ids = []
    for r in results:
        # robust extraction for various keys
        doc_id = r.get("doc_id") or r.get("document_id") or r.get("id")
        if doc_id:
            ids.append(doc_id)
    return ids


def evaluate(
    testset_path: Path = TESTSET_PATH,
    top_k_list: List[int] = [3, 4, 5],
    metadata_filter: dict | None = None,
    rerank: bool = False,
    compute_confidence: bool = True,
):
    """Run evaluation across multiple top_k settings and optional filters.

    Returns a dict mapping top_k -> (summary, per_query_results)
    """
    testset = RAGTestSet.load_from_file(str(testset_path))

    idx, docs, meta = load_retrieval_assets()

    # Validate that all referenced IDs exist in the vector store
    try:
        validate_testset_ids(testset, docs)
        print("[ID Validation]", confirm_ids(testset, docs))
    except Exception as e:
        # Fail fast with clear error listing missing ids
        print("[ID Validation ERROR]")
        raise

    results_by_k = {}

    for top_k in top_k_list:
        per_query_results = []

        for sample in testset.samples:
            q = sample.query
            relevant = sample.relevant_doc_ids
            retrieved = retrieve_top_k(q, idx, docs, meta, k=top_k, metadata_filter=metadata_filter)

            if rerank:
                retrieved = rerank_by_cosine(q, retrieved)

            retrieved_ids = _extract_doc_ids(retrieved)

            recall = metrics.recall_at_k(retrieved_ids, relevant, top_k)
            precision = metrics.precision_at_k(retrieved_ids, relevant, top_k)
            mrr = metrics.mean_reciprocal_rank(retrieved_ids, relevant)
            ndcg = metrics.ndcg_at_k(retrieved_ids, relevant, top_k)

            confidence = None
            if compute_confidence:
                confidence = max(0.0, min(1.0, 0.7 * recall + 0.3 * mrr))

            per_query_results.append(
                {
                    "query": q,
                    "relevant": relevant,
                    "retrieved": retrieved,
                    "retrieved_ids": retrieved_ids,
                    "recall": recall,
                    "precision": precision,
                    "mrr": mrr,
                    "ndcg": ndcg,
                    "confidence": confidence,
                }
            )

        # Aggregation
        recalls = [r["recall"] for r in per_query_results]
        precisions = [r["precision"] for r in per_query_results]
        mrrs = [r["mrr"] for r in per_query_results]
        ndcgs = [r["ndcg"] for r in per_query_results]
        confidences = [r["confidence"] for r in per_query_results if r["confidence"] is not None]

        summary = {
            "queries": len(per_query_results),
            "mean_recall@k": statistics.mean(recalls) if recalls else 0.0,
            "mean_precision@k": statistics.mean(precisions) if precisions else 0.0,
            "mean_mrr": statistics.mean(mrrs) if mrrs else 0.0,
            "mean_ndcg@k": statistics.mean(ndcgs) if ndcgs else 0.0,
            "mean_confidence": statistics.mean(confidences) if confidences else None,
        }

        # Print a clean summary per top_k
        print(f"\n=== RAG Retrieval Evaluation Summary (top_k={top_k}) ===")
        print(f"Queries evaluated: {summary['queries']}")
        print(f"Mean Recall@{top_k}: {summary['mean_recall@k']:.4f}")
        print(f"Mean Precision@{top_k}: {summary['mean_precision@k']:.4f}")
        print(f"Mean MRR: {summary['mean_mrr']:.4f}")
        print(f"Mean nDCG@{top_k}: {summary['mean_ndcg@k']:.4f}")
        if summary["mean_confidence"] is not None:
            print(f"Mean Retrieval Confidence: {summary['mean_confidence']:.4f}")

        # Precision-focused diagnostics
        print("\n=== Precision Diagnostics (Precision@k < 0.4) ===")
        for r in per_query_results:
            if r["precision"] < 0.4:
                # List retrieved but irrelevant doc_ids with scores
                irrelevant = [
                    (item.get("doc_id"), item.get("score", 0.0))
                    for item in r["retrieved"]
                    if item.get("doc_id") not in r["relevant"]
                ]
                if irrelevant:
                    print("- Query:", r["query"])
                    print("  Irrelevant retrieved (id, score):", irrelevant)

        # Stress-test diagnostics specifically for k==3 thresholds
        if top_k == 3:
            print("\n=== Stress-test Diagnostics (Recall@3 < 1.0 or Precision@3 < 0.6) ===")
            for r in per_query_results:
                if r["recall"] < 1.0 or r["precision"] < 0.6:
                    # retrieved doc ids and scores
                    retrieved_list = [(it.get("doc_id"), it.get("score", 0.0)) for it in r["retrieved"]]
                    missing = [d for d in r["relevant"] if d not in r["retrieved_ids"]]
                    print("- Query:", r["query"])
                    print("  Retrieved (id, score):", retrieved_list)
                    print("  Missing relevant ids:", missing)

        # Failure analysis: list failed queries with missing relevant ids
        print("\n=== Failure Analysis (Recall < 1.0) ===")
        failures = 0
        for r in per_query_results:
            if r["recall"] < 1.0:
                failures += 1
                missing = [d for d in r["relevant"] if d not in r["retrieved_ids"]]
                print("- Query:", r["query"])
                print("  Relevant (expected):", r["relevant"])
                print("  Retrieved (top-{}):".format(top_k), r["retrieved_ids"])
                print("  Missing relevant ids:", missing)

        if failures == 0:
            print("All queries retrieved at least one relevant document within top-K (Recall=1.0).")

        results_by_k[top_k] = (summary, per_query_results)

    # Regression guardrails w.r.t. thresholds
    print("\n=== Regression Guardrails ===")
    # Warn if Recall@5 < 0.90
    if 5 in results_by_k:
        recall5 = results_by_k[5][0].get("mean_recall@k", 0.0)
        if recall5 < 0.90:
            print(f"WARNING: Mean Recall@5 = {recall5:.4f} < 0.90")
    # Warn if Precision@3 < 0.70
    if 3 in results_by_k:
        prec3 = results_by_k[3][0].get("mean_precision@k", 0.0)
        if prec3 < 0.70:
            print(f"WARNING: Mean Precision@3 = {prec3:.4f} < 0.70")
    # Warn if MRR < 0.75 (use top_k=3 summary MRR if available)
    mrr_values = [results_by_k[k][0].get("mean_mrr", 0.0) for k in results_by_k]
    if mrr_values:
        overall_mrr = sum(mrr_values) / len(mrr_values)
        if overall_mrr < 0.75:
            print(f"WARNING: Mean MRR across evaluated K values = {overall_mrr:.4f} < 0.75")

    return results_by_k


if __name__ == "__main__":
    evaluate()
