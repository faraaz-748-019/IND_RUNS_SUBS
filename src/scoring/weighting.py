"""
Score weighting and fusion module.
Combines semantic, attribute, and behavioral scores into a final ranking score.
Weights are configurable via config.yaml — no code changes needed to tune.
"""


def compute_final_score(
    semantic_score: float,
    attribute_score: float,
    behavioral_score: float,
    weights: dict,
    honeypot_penalty: bool = False,
) -> float:
    """
    Compute the final weighted score for a candidate.
    
    Args:
        semantic_score: Score from hybrid retrieval + reranking (0-1)
        attribute_score: Score from structured attribute matching (0-1)
        behavioral_score: Score from behavioral signals (0-1)
        weights: Dict with keys 'semantic', 'attribute', 'behavioral'
        honeypot_penalty: If True, apply severe penalty
    
    Returns:
        Final score (0-1)
    """
    if honeypot_penalty:
        return 0.0  # Honeypots get zero — they must never appear in top-100
    
    final = (
        weights.get("semantic", 0.30) * semantic_score +
        weights.get("attribute", 0.45) * attribute_score +
        weights.get("behavioral", 0.25) * behavioral_score
    )
    
    return max(0.0, min(1.0, final))


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize a list of scores to 0-1 range."""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def rank_candidates(scored_candidates: list[dict]) -> list[dict]:
    """
    Sort candidates by final_score descending and assign ranks.
    Tie-break by candidate_id ascending (per submission spec).
    """
    sorted_cands = sorted(
        scored_candidates,
        key=lambda c: (-c.get("final_score", 0), c.get("candidate_id", "")),
    )
    
    for i, cand in enumerate(sorted_cands):
        cand["rank"] = i + 1
    
    return sorted_cands


if __name__ == "__main__":
    # Quick test
    weights = {"semantic": 0.30, "attribute": 0.45, "behavioral": 0.25}
    
    test_cases = [
        (0.8, 0.9, 0.7, False),  # Strong candidate
        (0.6, 0.3, 0.5, False),  # Weak on attributes
        (0.9, 0.8, 0.1, False),  # Low behavioral
        (0.9, 0.9, 0.9, True),   # Honeypot
    ]
    
    for sem, attr, beh, hp in test_cases:
        score = compute_final_score(sem, attr, beh, weights, hp)
        print(f"Sem={sem:.1f} Attr={attr:.1f} Beh={beh:.1f} HP={hp} → Final={score:.3f}")
