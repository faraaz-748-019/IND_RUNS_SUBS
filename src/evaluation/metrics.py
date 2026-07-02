"""
Information retrieval evaluation metrics.
Implements: NDCG@K, MAP, Precision@K, Recall@K, MRR.
Plus the competition's specific composite score formula.
"""

import math
import numpy as np
from typing import Optional


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Precision@K: fraction of top-K results that are relevant.
    """
    if k <= 0:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Recall@K: fraction of relevant items found in top-K.
    """
    if not relevant_ids:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def average_precision(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """
    Average Precision for a single query.
    AP = (1/|R|) * sum(P@k * rel(k)) for k=1..n
    """
    if not relevant_ids:
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    
    for i, rid in enumerate(ranked_ids):
        if rid in relevant_ids:
            hits += 1
            sum_precisions += hits / (i + 1)
    
    return sum_precisions / len(relevant_ids)


def mean_average_precision(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """
    MAP for a single query (same as AP when there's one query).
    """
    return average_precision(ranked_ids, relevant_ids)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """
    Reciprocal Rank: 1/rank of the first relevant result.
    """
    for i, rid in enumerate(ranked_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(ranked_ids: list[str], relevance_scores: dict, k: int) -> float:
    """
    Discounted Cumulative Gain @ K.
    Uses the formula: DCG = sum(rel_i / log2(i+2)) for i=0..k-1
    """
    dcg = 0.0
    for i, rid in enumerate(ranked_ids[:k]):
        rel = relevance_scores.get(rid, 0)
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0
    return dcg


def ndcg_at_k(ranked_ids: list[str], relevance_scores: dict, k: int) -> float:
    """
    Normalized DCG @ K.
    NDCG = DCG@K / IDCG@K where IDCG is the ideal DCG.
    """
    dcg = dcg_at_k(ranked_ids, relevance_scores, k)
    
    # Ideal ranking: sort by relevance descending
    ideal_order = sorted(relevance_scores.keys(), key=lambda x: relevance_scores[x], reverse=True)
    idcg = dcg_at_k(ideal_order, relevance_scores, k)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def competition_composite(
    ranked_ids: list[str],
    relevance_scores: dict,
    relevant_ids: set[str],
) -> dict:
    """
    Compute the competition's specific composite score:
    Final = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
    
    Args:
        ranked_ids: Ordered list of candidate IDs (rank 1 first)
        relevance_scores: Dict mapping candidate_id → relevance score (0-5)
        relevant_ids: Set of candidate_ids considered relevant (tier 3+)
    
    Returns:
        Dict with all metrics and the composite score
    """
    ndcg_10 = ndcg_at_k(ranked_ids, relevance_scores, 10)
    ndcg_50 = ndcg_at_k(ranked_ids, relevance_scores, 50)
    map_score = mean_average_precision(ranked_ids, relevant_ids)
    p_10 = precision_at_k(ranked_ids, relevant_ids, 10)
    p_5 = precision_at_k(ranked_ids, relevant_ids, 5)
    mrr = reciprocal_rank(ranked_ids, relevant_ids)
    
    composite = 0.50 * ndcg_10 + 0.30 * ndcg_50 + 0.15 * map_score + 0.05 * p_10
    
    return {
        "composite": round(composite, 4),
        "ndcg_10": round(ndcg_10, 4),
        "ndcg_50": round(ndcg_50, 4),
        "map": round(map_score, 4),
        "p_10": round(p_10, 4),
        "p_5": round(p_5, 4),
        "mrr": round(mrr, 4),
    }


def evaluate_ranking(
    ranked_ids: list[str],
    eval_set: dict,
) -> dict:
    """
    Evaluate a ranking against a labeled evaluation set.
    
    Args:
        ranked_ids: Ordered list of candidate IDs (rank 1 first)
        eval_set: Dict with:
            - relevance_scores: {candidate_id: score (0-5)}
            - relevant_threshold: minimum score to be "relevant" (default 3)
    
    Returns:
        Full metrics dict
    """
    relevance_scores = eval_set.get("relevance_scores", {})
    threshold = eval_set.get("relevant_threshold", 3)
    relevant_ids = {cid for cid, score in relevance_scores.items() if score >= threshold}
    
    return competition_composite(ranked_ids, relevance_scores, relevant_ids)


if __name__ == "__main__":
    # Quick test
    ranked = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    relevance = {"A": 5, "B": 0, "C": 4, "D": 3, "E": 0, "F": 2, "G": 5, "H": 1, "I": 4, "J": 0}
    relevant = {"A", "C", "D", "G", "I"}
    
    metrics = competition_composite(ranked, relevance, relevant)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
