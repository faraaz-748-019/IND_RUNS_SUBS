"""
Maximal Marginal Relevance (MMR) diversity re-ranking.
Prevents the shortlist from being dominated by near-duplicate profiles.
Gives recruiters a more useful spread of strong candidates.

MMR formula:
  MMR(d) = λ * Rel(d, q) - (1-λ) * max(Sim(d, d_selected))

Where λ controls the relevance-diversity tradeoff:
  λ = 1.0 → pure relevance (no diversity)
  λ = 0.0 → pure diversity (maximally different candidates)
"""

import numpy as np
from typing import Optional


def mmr_rerank(
    candidates: list[dict],
    embeddings: dict,
    lam: float = 0.8,
    top_k: int = 100,
) -> list[dict]:
    """
    Apply MMR diversity re-ranking to the candidate list.
    
    Args:
        candidates: Scored and ranked candidates (sorted by final_score)
        embeddings: Dict mapping candidate_id → embedding vector (numpy array)
        lam: Lambda parameter (0-1). Higher = more relevance, lower = more diversity.
        top_k: Number of candidates to select
    
    Returns:
        MMR-reranked list of candidates
    """
    if not candidates or lam >= 1.0:
        return candidates[:top_k]
    
    # Pool of candidates to select from
    pool = list(candidates[:min(len(candidates), top_k * 3)])
    selected = []
    selected_ids = set()
    
    # Normalize relevance scores
    max_score = max(c.get("final_score", 0) for c in pool) or 1.0
    
    for _ in range(min(top_k, len(pool))):
        best_mmr = -float("inf")
        best_idx = 0
        
        for i, cand in enumerate(pool):
            if cand["candidate_id"] in selected_ids:
                continue
            
            # Relevance term
            relevance = cand.get("final_score", 0) / max_score
            
            # Diversity term: max similarity to any already-selected candidate
            max_sim = 0.0
            cand_id = cand["candidate_id"]
            if cand_id in embeddings and selected:
                cand_emb = embeddings[cand_id]
                for sel in selected:
                    sel_id = sel["candidate_id"]
                    if sel_id in embeddings:
                        sim = _cosine_similarity(cand_emb, embeddings[sel_id])
                        max_sim = max(max_sim, sim)
            
            # MMR score
            mmr_score = lam * relevance - (1 - lam) * max_sim
            
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i
        
        selected_cand = pool[best_idx]
        selected.append(selected_cand)
        selected_ids.add(selected_cand["candidate_id"])
    
    # Re-assign ranks
    for i, cand in enumerate(selected):
        cand["rank"] = i + 1
    
    return selected


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


if __name__ == "__main__":
    print("MMR diversity re-ranking module ready.")
    print("Usage: mmr_rerank(candidates, embeddings, lambda=0.8, top_k=100)")
