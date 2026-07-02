#!/usr/bin/env python3
"""
Main ranking pipeline.
Produces the submission CSV from candidates.jsonl.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

The ranking step must complete within 5 minutes on CPU with 16 GB RAM.
Pre-computed embeddings and FAISS index are loaded from cache.
"""

import argparse
import csv
import json
import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="AI Candidate Ranking Engine")
    parser.add_argument("--candidates", required=True, help="Path to candidates file (.jsonl or .json)")
    parser.add_argument("--out", default="output/submission.csv", help="Output CSV path")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--top-k", type=int, default=100, help="Number of candidates to output")
    parser.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder reranking")
    parser.add_argument("--no-mmr", action="store_true", help="Skip MMR diversity reranking")
    parser.add_argument("--sample", type=int, default=None, help="Only process first N candidates")
    args = parser.parse_args()
    
    from src.ingestion.loader import (
        load_config, load_all_candidates, load_job_description,
        build_candidate_text,
    )
    from src.ingestion.jd_parser import parse_jd
    from src.ingestion.honeypot_detector import filter_honeypots
    from src.fairness.bias_masker import mask_candidate_for_scoring
    from src.retrieval.embeddings import (
        get_embedding_model, load_embeddings, load_faiss_index,
        compute_embeddings, build_faiss_index, search_faiss,
    )
    from src.retrieval.bm25_retriever import bm25_retrieve, build_bm25_index
    from src.retrieval.hybrid import reciprocal_rank_fusion
    from src.retrieval.reranker import get_reranker_model, rerank
    from src.scoring.attribute_scorer import compute_attribute_score
    from src.scoring.behavioral_scorer import compute_behavioral_score
    from src.scoring.weighting import compute_final_score, normalize_scores, rank_candidates
    from src.ranking.mmr import mmr_rerank
    from src.ranking.explainer import generate_rationale, generate_score_breakdown
    
    config = load_config(args.config)
    retrieval_config = config.get("retrieval", {})
    weights = config.get("weights", {})
    attr_weights = config.get("attribute_weights", {})
    
    print("=" * 60)
    print("AI CANDIDATE RANKING ENGINE")
    print("=" * 60)
    t_start = time.perf_counter()
    
    # ========================================================================
    # Step 1: Load JD and parse requirements
    # ========================================================================
    print("\n[1/9] Loading job description...")
    jd_text = load_job_description(config["paths"]["job_description"])
    jd = parse_jd(jd_text, config)
    print(f"    JD: {jd['title']} at {jd['company']}")
    print(f"    Must-have skills: {len(jd['must_have_skills'])}")
    print(f"    Experience range: {jd['experience_range']}")
    
    # ========================================================================
    # Step 2: Load candidates
    # ========================================================================
    print("\n[2/9] Loading candidates...")
    t1 = time.perf_counter()
    candidates = load_all_candidates(args.candidates, max_candidates=args.sample)
    print(f"    Loaded {len(candidates):,} candidates ({time.perf_counter() - t1:.1f}s)")
    
    # ========================================================================
    # Step 3: Honeypot detection
    # ========================================================================
    print("\n[3/9] Detecting honeypots...")
    clean_candidates, honeypots = filter_honeypots(candidates, config)
    print(f"    Clean: {len(clean_candidates):,} | Honeypots detected: {len(honeypots):,}")
    
    # ========================================================================
    # Step 4: Build candidate text representations
    # ========================================================================
    print("\n[4/9] Building text representations...")
    candidate_texts = {}
    for c in clean_candidates:
        masked = mask_candidate_for_scoring(c)
        candidate_texts[c["candidate_id"]] = build_candidate_text(masked)
    
    cand_ids = list(candidate_texts.keys())
    cand_text_list = [candidate_texts[cid] for cid in cand_ids]
    
    # ========================================================================
    # Step 5: Hybrid retrieval (BM25 + Dense)
    # ========================================================================
    print("\n[5/9] Hybrid retrieval...")
    t2 = time.perf_counter()
    
    # Try loading pre-computed embeddings and FAISS index
    embeddings_cache = load_embeddings(config["paths"]["embedding_cache"])
    faiss_index_cache = load_faiss_index(config["paths"]["faiss_index"])
    
    # Load pre-computed candidate IDs mapping if available
    ids_cache_path = config["paths"]["candidate_ids_cache"]
    if os.path.exists(ids_cache_path):
        with open(ids_cache_path, "r") as f:
            cached_ids = json.load(f)
    else:
        cached_ids = None
    
    # Embedding model (needed for query embedding even with cache)
    embedding_model = get_embedding_model(config["models"]["embedding"])
    
    if embeddings_cache is not None and faiss_index_cache is not None and cached_ids:
        print("    Using pre-computed embeddings and FAISS index")
        # Compute query embedding
        query_emb = compute_embeddings(
            [jd.get("embedding_text", jd_text)],
            embedding_model, show_progress=False,
        )
        # Dense retrieval from cache
        scores, indices = search_faiss(
            query_emb, faiss_index_cache,
            top_k=retrieval_config.get("dense_top_k", 1000),
        )
        dense_results = []
        for score, idx in zip(scores, indices):
            if 0 <= idx < len(cached_ids):
                cid = cached_ids[idx]
                if cid in candidate_texts:  # Only include non-honeypots
                    dense_results.append((cid, float(score)))
    else:
        print("    Computing embeddings on-the-fly...")
        from src.retrieval.embeddings import dense_retrieve
        dense_results = dense_retrieve(
            jd.get("embedding_text", jd_text),
            cand_text_list, cand_ids,
            model=embedding_model,
            top_k=retrieval_config.get("dense_top_k", 1000),
        )
    
    # BM25 retrieval
    print("    Running BM25...")
    bm25_index = build_bm25_index(cand_text_list)
    bm25_results = bm25_retrieve(
        jd.get("embedding_text", jd_text),
        cand_text_list, cand_ids,
        top_k=retrieval_config.get("bm25_top_k", 1000),
        bm25_index=bm25_index,
    )
    
    # RRF fusion
    print("    Fusing with RRF...")
    hybrid_results = reciprocal_rank_fusion(
        [dense_results, bm25_results],
        k=retrieval_config.get("rrf_k", 60),
    )
    hybrid_top = hybrid_results[:retrieval_config.get("hybrid_top_k", 500)]
    
    print(f"    Hybrid pool: {len(hybrid_top)} candidates ({time.perf_counter() - t2:.1f}s)")
    
    # ========================================================================
    # Step 6: Cross-encoder reranking
    # ========================================================================
    if not args.no_rerank:
        print("\n[6/9] Cross-encoder reranking...")
        t3 = time.perf_counter()
        reranker = get_reranker_model(config["models"]["cross_encoder"])
        reranked = rerank(
            jd.get("embedding_text", jd_text),
            hybrid_top,
            candidate_texts,
            reranker,
            top_k=retrieval_config.get("rerank_top_k", 300),
        )
        # Normalize reranker scores
        if reranked:
            min_s = min(s for _, s in reranked)
            max_s = max(s for _, s in reranked)
            rng = max_s - min_s if max_s != min_s else 1.0
            semantic_scores = {cid: (s - min_s) / rng for cid, s in reranked}
        else:
            semantic_scores = {}
        print(f"    Reranked {len(reranked)} candidates ({time.perf_counter() - t3:.1f}s)")
    else:
        print("\n[6/9] Skipping cross-encoder reranking (--no-rerank)")
        # Use hybrid scores as semantic scores
        max_h = max(s for _, s in hybrid_top) if hybrid_top else 1.0
        semantic_scores = {cid: s / max_h for cid, s in hybrid_top}
    
    # ========================================================================
    # Step 7: Full scoring (attribute + behavioral + semantic fusion)
    # ========================================================================
    print("\n[7/9] Computing attribute and behavioral scores...")
    t4 = time.perf_counter()
    
    # Only score candidates in the retrieval pool
    pool_ids = set(semantic_scores.keys())
    pool_candidates = [c for c in clean_candidates if c["candidate_id"] in pool_ids]
    
    scored = []
    for candidate in pool_candidates:
        cid = candidate["candidate_id"]
        sem_score = semantic_scores.get(cid, 0.0)
        attr_result = compute_attribute_score(candidate, jd, attr_weights)
        beh_result = compute_behavioral_score(candidate, config)
        
        final = compute_final_score(
            sem_score, attr_result["score"], beh_result["score"], weights,
        )
        
        scored.append({
            "candidate_id": cid,
            "candidate": candidate,
            "final_score": final,
            "semantic_score": sem_score,
            "semantic_score_normalized": sem_score,
            "attribute_score": attr_result["score"],
            "behavioral_score": beh_result["score"],
            "attribute_details": attr_result["sub_scores"],
            "behavioral_details": beh_result["sub_scores"],
            "is_honeypot": False,
        })
    
    scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    print(f"    Scored {len(scored)} candidates ({time.perf_counter() - t4:.1f}s)")
    
    # ========================================================================
    # Step 8: MMR diversity re-ranking
    # ========================================================================
    mmr_config = config.get("mmr", {})
    if mmr_config.get("enabled", True) and not args.no_mmr:
        print("\n[8/9] MMR diversity re-ranking...")
        # Build embedding lookup for MMR
        emb_lookup = {}
        if embeddings_cache is not None and cached_ids:
            id_to_idx = {cid: i for i, cid in enumerate(cached_ids)}
            for s in scored[:300]:
                cid = s["candidate_id"]
                if cid in id_to_idx:
                    emb_lookup[cid] = embeddings_cache[id_to_idx[cid]]
        
        final_ranked = mmr_rerank(
            scored, emb_lookup,
            lam=mmr_config.get("lambda", 0.8),
            top_k=args.top_k,
        )
    else:
        print("\n[8/9] Skipping MMR diversity re-ranking")
        final_ranked = scored[:args.top_k]
        for i, c in enumerate(final_ranked):
            c["rank"] = i + 1
    
    # ========================================================================
    # Step 9: Generate rationales and write output
    # ========================================================================
    print(f"\n[9/9] Generating rationales and writing output...")
    
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    
    # Write CSV (submission format)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for cand in final_ranked[:args.top_k]:
            rationale = generate_rationale(cand)
            writer.writerow([
                cand["candidate_id"],
                cand["rank"],
                f"{cand['final_score']:.4f}",
                rationale,
            ])
    
    print(f"    Wrote {min(len(final_ranked), args.top_k)} candidates to {args.out}")
    
    # Write detailed JSON output
    detailed_path = config["paths"].get("output_detailed", "output/submission_detailed.json")
    os.makedirs(os.path.dirname(detailed_path) or ".", exist_ok=True)
    
    detailed_output = []
    for cand in final_ranked[:args.top_k]:
        breakdown = generate_score_breakdown(cand)
        detailed_output.append({
            "candidate_id": cand["candidate_id"],
            "rank": cand["rank"],
            "final_score": cand["final_score"],
            "reasoning": generate_rationale(cand),
            "score_breakdown": breakdown,
            "profile_summary": {
                "title": cand["candidate"].get("profile", {}).get("current_title", ""),
                "company": cand["candidate"].get("profile", {}).get("current_company", ""),
                "experience": cand["candidate"].get("profile", {}).get("years_of_experience", 0),
                "location": cand["candidate"].get("profile", {}).get("location", ""),
            },
        })
    
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_output, f, indent=2, default=str)
    
    print(f"    Detailed output: {detailed_path}")
    
    # Summary
    total_time = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"RANKING COMPLETE in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 60}")
    print(f"\nTop 5 candidates:")
    for c in final_ranked[:5]:
        profile = c["candidate"].get("profile", {})
        print(f"  #{c['rank']} {c['candidate_id']} "
              f"({profile.get('current_title', '?')}, "
              f"{profile.get('years_of_experience', 0):.1f} yrs) "
              f"Score: {c['final_score']:.4f}")
    
    # Honeypot check
    hp_in_top100 = sum(1 for c in final_ranked[:100] if c.get("is_honeypot", False))
    print(f"\nHoneypots in top-100: {hp_in_top100} "
          f"({'SAFE' if hp_in_top100 == 0 else 'WARNING'})")
    
    if total_time > 300:
        print(f"\n[WARNING] Runtime ({total_time:.0f}s) exceeds 5-minute limit!")
    else:
        print(f"\n[OK] Runtime within 5-minute limit ({total_time:.0f}s / 300s)")


if __name__ == "__main__":
    main()
