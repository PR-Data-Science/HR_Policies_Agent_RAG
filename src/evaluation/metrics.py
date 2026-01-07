"""Pure, unit-testable retrieval metric implementations.

Provides: recall_at_k, precision_at_k, mean_reciprocal_rank, ndcg_at_k.
"""

from typing import List
import math


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Recall@K = (# relevant retrieved in top-K) / (# relevant total)

    Returns 0.0 if there are no relevant_ids.
    """
    if not relevant_ids:
        return 0.0
    topk = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in topk if doc_id in relevant_set)
    return hits / len(relevant_set)


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Precision@K = (# relevant retrieved in top-K) / K

    If k is 0, returns 0.0 to avoid division by zero.
    """
    if k <= 0:
        return 0.0
    topk = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in topk if doc_id in relevant_set)
    return hits / k


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """MRR for a single query: reciprocal of the rank of the first relevant item.

    If none of the retrieved_ids are relevant, returns 0.0.
    """
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / i
    return 0.0


def _dcg(scores: List[float]) -> float:
    """Compute DCG given a list of relevance scores (ordered by rank)."""
    dcg = 0.0
    for i, rel in enumerate(scores, start=1):
        denom = math.log2(i + 1)
        dcg += rel / denom
    return dcg


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """nDCG@K where relevance is binary (1 if relevant, 0 otherwise).

    nDCG = DCG / IDCG where IDCG is ideal DCG (all relevant docs ranked on top).
    Returns 0.0 if there are no relevant_ids.
    """
    if not relevant_ids:
        return 0.0
    topk = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    gains = [1.0 if doc_id in relevant_set else 0.0 for doc_id in topk]
    dcg = _dcg(gains)

    # Ideal gains: put as many relevant docs as possible at top
    ideal_rels = [1.0] * min(len(relevant_set), k)
    idcg = _dcg(ideal_rels)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
