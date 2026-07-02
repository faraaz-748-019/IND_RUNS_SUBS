"""
Hybrid retrieval: combines BM25 (sparse) and dense embedding (dense) retrieval.
Uses Reciprocal Rank Fusion (RRF) for score combination.
This is the key differentiator — most teams will only do one or the other.
"""

from typing import Optional
import numpy as np


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    
    RRF is a simple, robust method for combining heterogeneous rankings.
    For each candidate, the fused score is: sum(1 / (k + rank_i)) across all lists.
    
    Args:
        ranked_lists: List of ranked result lists, each containing (candidate_id, score) tuples
        k: RRF constant (default 60, from the original paper)
    
    Returns:
        Fused ranked list of (candidate_id, rrf_score) tuples
    """
    rrf_scores = {}
    
    for ranked_list in ranked_lists:
        for rank, (candidate_id, _score) in enumerate(ranked_list):
            if candidate_id not in rrf_scores:
                rrf_scores[candidate_id] = 0.0
            rrf_scores[candidate_id] += 1.0 / (k + rank + 1)
    
    # Sort by fused score
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def linear_combination(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    alpha: float = 0.5,
) -> list[tuple[str, float]]:
    """
    Combine dense and sparse scores with linear interpolation.
    
    final_score = alpha * normalized_dense + (1-alpha) * normalized_sparse
    
    Args:
        dense_results: Dense retrieval results
        sparse_results: Sparse (BM25) retrieval results
        alpha: Weight for dense scores (0-1)
    
    Returns:
        Combined ranked list
    """
    # Normalize scores to 0-1 range
    dense_dict = _normalize_scores({cid: s for cid, s in dense_results})
    sparse_dict = _normalize_scores({cid: s for cid, s in sparse_results})
    
    # Combine
    all_ids = set(dense_dict.keys()) | set(sparse_dict.keys())
    combined = {}
    for cid in all_ids:
        d_score = dense_dict.get(cid, 0.0)
        s_score = sparse_dict.get(cid, 0.0)
        combined[cid] = alpha * d_score + (1 - alpha) * s_score
    
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize scores to 0-1 range."""
    if not scores:
        return {}
    values = list(scores.values())
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return {k: 0.5 for k in scores}
    return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}


def hybrid_retrieve(
    query_text: str,
    candidate_texts: list[str],
    candidate_ids: list[str],
    embedding_model,
    config: dict,
    embeddings_cache=None,
    faiss_index_cache=None,
    bm25_index_cache=None,
) -> list[tuple[str, float]]:
    """
    Full hybrid retrieval pipeline: BM25 + Dense + RRF fusion.
    
    Args:
        query_text: Job description text
        candidate_texts: List of candidate text representations
        candidate_ids: Corresponding candidate IDs
        embedding_model: SentenceTransformer model
        config: Master configuration
        embeddings_cache: Pre-computed embeddings
        faiss_index_cache: Pre-built FAISS index
        bm25_index_cache: Pre-built BM25 index
    
    Returns:
        Hybrid ranked list of (candidate_id, score) tuples
    """
    from src.retrieval.embeddings import dense_retrieve
    from src.retrieval.bm25_retriever import bm25_retrieve
    
    retrieval_config = config.get("retrieval", {})
    bm25_top_k = retrieval_config.get("bm25_top_k", 1000)
    dense_top_k = retrieval_config.get("dense_top_k", 1000)
    hybrid_top_k = retrieval_config.get("hybrid_top_k", 500)
    rrf_k = retrieval_config.get("rrf_k", 60)
    
    print("  Running BM25 retrieval...")
    sparse_results = bm25_retrieve(
        query_text, candidate_texts, candidate_ids,
        top_k=bm25_top_k, bm25_index=bm25_index_cache,
    )
    
    print("  Running dense retrieval...")
    dense_results = dense_retrieve(
        query_text, candidate_texts, candidate_ids,
        model=embedding_model, top_k=dense_top_k,
        embeddings_cache=embeddings_cache, index_cache=faiss_index_cache,
    )
    
    print("  Fusing with Reciprocal Rank Fusion...")
    hybrid_results = reciprocal_rank_fusion(
        [dense_results, sparse_results], k=rrf_k,
    )
    
    return hybrid_results[:hybrid_top_k]


if __name__ == "__main__":
    # Quick test
    dense = [("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6)]
    sparse = [("B", 5.0), ("C", 4.0), ("A", 3.0), ("E", 2.0)]
    
    rrf = reciprocal_rank_fusion([dense, sparse])
    print("RRF results:")
    for cid, score in rrf[:5]:
        print(f"  {cid}: {score:.6f}")
    
    linear = linear_combination(dense, sparse, alpha=0.5)
    print("\nLinear combination (alpha=0.5):")
    for cid, score in linear[:5]:
        print(f"  {cid}: {score:.6f}")
