"""
Cross-encoder reranker.
After hybrid retrieval narrows to ~300-500 candidates, the cross-encoder
rescores (query, candidate) pairs for higher precision at the top.
Uses free, CPU-friendly cross-encoder model.
"""

from typing import Optional
import numpy as np


def get_reranker_model(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Load a cross-encoder reranking model.
    CPU-friendly, no API needed.
    """
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def rerank(
    query_text: str,
    candidates: list[tuple[str, float]],
    candidate_texts: dict[str, str],
    reranker_model,
    top_k: int = 300,
    batch_size: int = 64,
) -> list[tuple[str, float]]:
    """
    Rerank candidates using a cross-encoder model.
    
    Cross-encoders are more accurate than bi-encoders because they
    process the (query, document) pair together, allowing full attention
    between query and document tokens. The tradeoff is speed — hence
    we only rerank the top candidates from hybrid retrieval.
    
    Args:
        query_text: Job description text
        candidates: List of (candidate_id, initial_score) from hybrid retrieval
        candidate_texts: Dict mapping candidate_id to text representation
        reranker_model: CrossEncoder model
        top_k: Number of candidates to rerank (and return)
        batch_size: Batch size for cross-encoder inference
    
    Returns:
        Reranked list of (candidate_id, cross_encoder_score) tuples
    """
    # Limit to top_k candidates for reranking
    candidates_to_rerank = candidates[:top_k]
    
    # Build (query, candidate_text) pairs
    pairs = []
    valid_ids = []
    for cid, _score in candidates_to_rerank:
        if cid in candidate_texts:
            # Truncate candidate text to avoid exceeding model's max length
            cand_text = candidate_texts[cid][:1500]
            pairs.append([query_text[:500], cand_text])
            valid_ids.append(cid)
    
    if not pairs:
        return candidates_to_rerank
    
    # Score with cross-encoder
    print(f"  Cross-encoder reranking {len(pairs)} candidates...")
    scores = reranker_model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    
    # Combine with results
    reranked = list(zip(valid_ids, [float(s) for s in scores]))
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    return reranked


if __name__ == "__main__":
    print("Cross-encoder reranker module loaded.")
    print("Model: cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("Use case: precision boost for top-N candidates after hybrid retrieval")
