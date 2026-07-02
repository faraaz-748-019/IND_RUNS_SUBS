"""
Dense embedding retrieval using sentence-transformers + FAISS.
Pre-computes embeddings and builds ANN index for fast similarity search.
All models are free, open-source, and CPU-friendly.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load a sentence-transformer embedding model.
    Uses CPU-friendly models that run locally — no API calls.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def compute_embeddings(
    texts: list[str],
    model,
    batch_size: int = 128,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        model: SentenceTransformer model
        batch_size: Encoding batch size
        show_progress: Show progress bar
    
    Returns:
        numpy array of shape (n_texts, embedding_dim)
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # L2 normalize for cosine similarity via dot product
    )
    return np.array(embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = False):
    """
    Build a FAISS index for approximate nearest neighbor search.
    Uses IndexFlatIP (inner product) since embeddings are L2-normalized.
    
    For 100K candidates with 384-dim embeddings, this is fast enough
    with flat index. For larger datasets, switch to IVF or HNSW.
    """
    import faiss
    
    dim = embeddings.shape[1]
    n = embeddings.shape[0]
    
    if n <= 50000:
        # Flat index — exact search, fast enough for moderate datasets
        index = faiss.IndexFlatIP(dim)
    else:
        # IVF index for larger datasets — approximate but much faster
        nlist = min(int(np.sqrt(n)), 256)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.nprobe = min(nlist, 20)  # search 20 clusters
    
    index.add(embeddings)
    return index


def search_faiss(
    query_embedding: np.ndarray,
    index,
    top_k: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS index for nearest neighbors.
    
    Args:
        query_embedding: Query vector (1, dim) or (dim,)
        index: FAISS index
        top_k: Number of results to return
    
    Returns:
        (scores, indices) tuple
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    scores, indices = index.search(query_embedding, top_k)
    return scores[0], indices[0]


def save_embeddings(embeddings: np.ndarray, filepath: str):
    """Save embeddings to disk for caching."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, embeddings)


def load_embeddings(filepath: str) -> Optional[np.ndarray]:
    """Load cached embeddings from disk."""
    if os.path.exists(filepath):
        return np.load(filepath)
    return None


def save_faiss_index(index, filepath: str):
    """Save FAISS index to disk."""
    import faiss
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    faiss.write_index(index, filepath)


def load_faiss_index(filepath: str):
    """Load FAISS index from disk."""
    import faiss
    if os.path.exists(filepath):
        return faiss.read_index(filepath)
    return None


def dense_retrieve(
    query_text: str,
    candidate_texts: list[str],
    candidate_ids: list[str],
    model,
    top_k: int = 1000,
    embeddings_cache: Optional[np.ndarray] = None,
    index_cache=None,
) -> list[tuple[str, float]]:
    """
    Full dense retrieval pipeline: embed query, search index, return results.
    
    Args:
        query_text: Job description text
        candidate_texts: List of candidate text representations
        candidate_ids: Corresponding candidate IDs
        model: SentenceTransformer model
        top_k: Number of results
        embeddings_cache: Pre-computed candidate embeddings
        index_cache: Pre-built FAISS index
    
    Returns:
        List of (candidate_id, score) tuples, sorted by score descending
    """
    # Embed query
    query_emb = compute_embeddings([query_text], model, show_progress=False)
    
    # Use cached or compute candidate embeddings
    if embeddings_cache is not None:
        candidate_embs = embeddings_cache
    else:
        candidate_embs = compute_embeddings(candidate_texts, model)
    
    # Use cached or build FAISS index
    if index_cache is not None:
        index = index_cache
    else:
        index = build_faiss_index(candidate_embs)
    
    # Search
    scores, indices = search_faiss(query_emb, index, top_k=top_k)
    
    # Map back to candidate IDs
    results = []
    for score, idx in zip(scores, indices):
        if idx >= 0 and idx < len(candidate_ids):
            results.append((candidate_ids[idx], float(score)))
    
    return results


if __name__ == "__main__":
    print("Dense embedding retrieval module loaded successfully.")
    print("Models used: all-MiniLM-L6-v2 (384-dim, CPU-friendly)")
    print("Index: FAISS (flat for <50K, IVF for larger)")
