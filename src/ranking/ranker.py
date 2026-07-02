"""
Final ranking engine.
Orchestrates the full scoring pipeline and produces the final ranked list.
"""

import sys
import os
from typing import Optional


def build_final_ranking(
    candidates: list[dict],
    semantic_scores: dict,
    jd: dict,
    config: dict,
) -> list[dict]:
    """
    Build the full ranking by combining all score components.
    
    Args:
        candidates: List of candidate dicts (already filtered for honeypots)
        semantic_scores: Dict mapping candidate_id → semantic score (from retrieval)
        jd: Parsed JD requirements
        config: Master configuration
    
    Returns:
        List of scored candidate dicts, sorted by final_score descending
    """
    from src.scoring.attribute_scorer import compute_attribute_score
    from src.scoring.behavioral_scorer import compute_behavioral_score
    from src.scoring.weighting import compute_final_score, normalize_scores, rank_candidates
    
    weights = config.get("weights", {})
    attr_weights = config.get("attribute_weights", {})
    
    scored = []
    
    for candidate in candidates:
        cid = candidate.get("candidate_id", "")
        
        # Semantic score (from hybrid retrieval + reranking)
        sem_score = semantic_scores.get(cid, 0.0)
        
        # Attribute score
        attr_result = compute_attribute_score(candidate, jd, attr_weights)
        
        # Behavioral score
        beh_result = compute_behavioral_score(candidate, config)
        
        # Check honeypot status
        hp_check = candidate.get("_honeypot_check", {})
        is_honeypot = hp_check.get("is_honeypot", False)
        
        # Final score
        final = compute_final_score(
            sem_score,
            attr_result["score"],
            beh_result["score"],
            weights,
            honeypot_penalty=is_honeypot,
        )
        
        scored.append({
            "candidate_id": cid,
            "candidate": candidate,
            "final_score": final,
            "semantic_score": sem_score,
            "attribute_score": attr_result["score"],
            "behavioral_score": beh_result["score"],
            "attribute_details": attr_result["sub_scores"],
            "behavioral_details": beh_result["sub_scores"],
            "is_honeypot": is_honeypot,
        })
    
    # Normalize semantic scores across the pool
    sem_vals = [s["semantic_score"] for s in scored]
    norm_sem = normalize_scores(sem_vals)
    for i, s in enumerate(scored):
        s["semantic_score_normalized"] = norm_sem[i]
        # Recompute final with normalized semantic
        s["final_score"] = compute_final_score(
            norm_sem[i],
            s["attribute_score"],
            s["behavioral_score"],
            weights,
            honeypot_penalty=s["is_honeypot"],
        )
    
    # Sort and assign ranks
    ranked = rank_candidates(scored)
    
    return ranked


if __name__ == "__main__":
    print("Ranking engine ready.")
