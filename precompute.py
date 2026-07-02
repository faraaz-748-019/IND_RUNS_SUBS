#!/usr/bin/env python3
"""
Pre-computation script.
Generates embeddings and FAISS index for all candidates.
This step can exceed the 5-minute compute window.
Run once, then rank.py loads from cache.

Usage:
    python precompute.py [--config config.yaml] [--candidates path/to/candidates.jsonl]
"""

import argparse
import json
import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings and FAISS index")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--candidates", default=None, help="Override candidates file path")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size")
    args = parser.parse_args()
    
    from src.ingestion.loader import load_config, stream_candidates, build_candidate_text
    from src.retrieval.embeddings import (
        get_embedding_model, compute_embeddings,
        build_faiss_index, save_embeddings, save_faiss_index,
    )
    
    config = load_config(args.config)
    candidates_path = args.candidates or config["paths"]["candidates"]
    
    print("=" * 60)
    print("PRE-COMPUTATION: Embeddings + FAISS Index")
    print("=" * 60)
    
    # Step 1: Load candidates and build text representations
    print("\n[1/4] Loading candidates and building text representations...")
    t0 = time.perf_counter()
    
    candidate_ids = []
    candidate_texts = []
    
    for i, candidate in enumerate(stream_candidates(candidates_path)):
        cid = candidate.get("candidate_id", f"UNK_{i}")
        text = build_candidate_text(candidate)
        candidate_ids.append(cid)
        candidate_texts.append(text)
        
        if (i + 1) % 10000 == 0:
            print(f"    Loaded {i + 1:,} candidates...")
    
    print(f"    Total: {len(candidate_ids):,} candidates ({time.perf_counter() - t0:.1f}s)")
    
    # Step 2: Save candidate IDs mapping
    print("\n[2/4] Saving candidate ID mapping...")
    ids_path = config["paths"]["candidate_ids_cache"]
    os.makedirs(os.path.dirname(ids_path), exist_ok=True)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(candidate_ids, f)
    
    # Also save candidate texts for BM25 and reranking
    texts_path = os.path.join(os.path.dirname(ids_path), "candidate_texts.json")
    # Save as a mapping for quick lookup
    text_map = dict(zip(candidate_ids, candidate_texts))
    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump(text_map, f)
    print(f"    Saved {len(candidate_ids):,} IDs and texts")
    
    # Step 3: Compute embeddings
    print(f"\n[3/4] Computing embeddings with {config['models']['embedding']}...")
    t1 = time.perf_counter()
    
    model = get_embedding_model(config["models"]["embedding"])
    embeddings = compute_embeddings(candidate_texts, model, batch_size=args.batch_size)
    
    emb_path = config["paths"]["embedding_cache"]
    save_embeddings(embeddings, emb_path)
    print(f"    Shape: {embeddings.shape} ({time.perf_counter() - t1:.1f}s)")
    print(f"    Saved to: {emb_path}")
    
    # Step 4: Build FAISS index
    print("\n[4/4] Building FAISS index...")
    t2 = time.perf_counter()
    
    index = build_faiss_index(embeddings)
    idx_path = config["paths"]["faiss_index"]
    save_faiss_index(index, idx_path)
    print(f"    Index size: {index.ntotal:,} vectors ({time.perf_counter() - t2:.1f}s)")
    print(f"    Saved to: {idx_path}")
    
    total_time = time.perf_counter() - t0
    print(f"\n{'=' * 60}")
    print(f"Pre-computation complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
