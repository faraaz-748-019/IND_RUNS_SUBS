"""
BM25 sparse keyword retrieval.
Provides keyword-based scoring as the sparse component of hybrid retrieval.
Uses rank_bm25 library (free, open-source).
"""

import re
import string
from typing import Optional


def tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer.
    Lowercases, removes punctuation, splits on whitespace.
    """
    text = text.lower()
    # Remove punctuation except hyphens (for terms like "A/B", "scikit-learn")
    text = re.sub(r'[^\w\s/\-]', ' ', text)
    tokens = text.split()
    # Filter very short tokens
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


def build_bm25_index(corpus_texts: list[str]):
    """
    Build a BM25 index from a corpus of candidate texts.
    
    Args:
        corpus_texts: List of text strings (one per candidate)
    
    Returns:
        BM25Okapi index object
    """
    from rank_bm25 import BM25Okapi
    
    tokenized_corpus = [tokenize(text) for text in corpus_texts]
    return BM25Okapi(tokenized_corpus)


def bm25_retrieve(
    query_text: str,
    candidate_texts: list[str],
    candidate_ids: list[str],
    top_k: int = 1000,
    bm25_index=None,
) -> list[tuple[str, float]]:
    """
    Retrieve top-K candidates using BM25 keyword scoring.
    
    Args:
        query_text: Job description or query text
        candidate_texts: List of candidate text representations
        candidate_ids: Corresponding candidate IDs
        top_k: Number of results
        bm25_index: Pre-built BM25 index (optional)
    
    Returns:
        List of (candidate_id, bm25_score) tuples, sorted by score descending
    """
    if bm25_index is None:
        bm25_index = build_bm25_index(candidate_texts)
    
    query_tokens = tokenize(query_text)
    scores = bm25_index.get_scores(query_tokens)
    
    # Get top-K indices
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append((candidate_ids[idx], float(scores[idx])))
    
    return results


if __name__ == "__main__":
    # Quick test
    corpus = [
        "Senior ML engineer with experience in NLP and information retrieval",
        "Marketing manager with 10 years in brand management",
        "Data scientist specializing in embeddings and vector databases FAISS",
    ]
    ids = ["CAND_001", "CAND_002", "CAND_003"]
    query = "ML engineer NLP embeddings retrieval FAISS"
    
    results = bm25_retrieve(query, corpus, ids)
    print("BM25 Results:")
    for cid, score in results:
        print(f"  {cid}: {score:.4f}")
