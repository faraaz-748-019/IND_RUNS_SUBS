"""
Ablation study runner.
Compares ranking quality across different pipeline configurations:
  (a) Keyword-only baseline (BM25)
  (b) Embedding-only
  (c) Hybrid without behavioral signals
  (d) Full pipeline
Presents results as a comparison table.
"""

import json
import os
from typing import Optional


def run_ablation(
    candidates: list[dict],
    jd: dict,
    config: dict,
    eval_set: dict,
    candidate_texts: dict,
    embedding_model=None,
    embeddings_cache=None,
    faiss_index_cache=None,
) -> dict:
    """
    Run ablation study comparing different pipeline configurations.
    
    Returns:
        Dict with results for each configuration
    """
    from src.evaluation.metrics import evaluate_ranking
    from src.scoring.attribute_scorer import compute_attribute_score
    from src.scoring.behavioral_scorer import compute_behavioral_score
    from src.scoring.weighting import compute_final_score, rank_candidates
    
    results = {}
    
    # Build candidate lookup
    cand_lookup = {c["candidate_id"]: c for c in candidates}
    cand_ids = list(candidate_texts.keys())
    cand_text_list = [candidate_texts[cid] for cid in cand_ids]
    jd_text = jd.get("embedding_text", "")
    
    attr_weights = config.get("attribute_weights", {})
    
    # ---- (a) Keyword-only (BM25) ----
    print("  Ablation: BM25 keyword-only...")
    try:
        from src.retrieval.bm25_retriever import bm25_retrieve
        bm25_results = bm25_retrieve(jd_text, cand_text_list, cand_ids, top_k=100)
        bm25_ranked = [cid for cid, _ in bm25_results]
        results["keyword_only"] = evaluate_ranking(bm25_ranked, eval_set)
        results["keyword_only"]["method"] = "BM25 keyword-only"
    except Exception as e:
        results["keyword_only"] = {"error": str(e), "method": "BM25 keyword-only"}
    
    # ---- (b) Embedding-only ----
    print("  Ablation: Embedding-only...")
    try:
        from src.retrieval.embeddings import dense_retrieve
        dense_results = dense_retrieve(
            jd_text, cand_text_list, cand_ids,
            model=embedding_model, top_k=100,
            embeddings_cache=embeddings_cache, index_cache=faiss_index_cache,
        )
        dense_ranked = [cid for cid, _ in dense_results]
        results["embedding_only"] = evaluate_ranking(dense_ranked, eval_set)
        results["embedding_only"]["method"] = "Embedding-only (dense)"
    except Exception as e:
        results["embedding_only"] = {"error": str(e), "method": "Embedding-only (dense)"}
    
    # ---- (c) Hybrid without behavioral signals ----
    print("  Ablation: Hybrid (no behavioral)...")
    try:
        from src.retrieval.hybrid import reciprocal_rank_fusion
        if "keyword_only" not in results.get("keyword_only", {}).get("error", "x"):
            hybrid_fused = reciprocal_rank_fusion(
                [[(cid, s) for cid, s in dense_results],
                 [(cid, s) for cid, s in bm25_results]],
                k=60,
            )
        else:
            hybrid_fused = dense_results  # fallback
        
        # Score with attributes only (no behavioral)
        scored = []
        hybrid_scores = {cid: score for cid, score in hybrid_fused}
        max_sem = max(hybrid_scores.values()) if hybrid_scores else 1.0
        
        for cid, sem_score in list(hybrid_fused)[:200]:
            if cid in cand_lookup:
                attr = compute_attribute_score(cand_lookup[cid], jd, attr_weights)
                norm_sem = sem_score / max_sem if max_sem > 0 else 0
                final = 0.40 * norm_sem + 0.60 * attr["score"]
                scored.append({"candidate_id": cid, "final_score": final})
        
        scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
        hybrid_no_beh_ranked = [s["candidate_id"] for s in scored[:100]]
        results["hybrid_no_behavioral"] = evaluate_ranking(hybrid_no_beh_ranked, eval_set)
        results["hybrid_no_behavioral"]["method"] = "Hybrid (no behavioral signals)"
    except Exception as e:
        results["hybrid_no_behavioral"] = {"error": str(e), "method": "Hybrid (no behavioral)"}
    
    # ---- (d) Full pipeline ----
    print("  Ablation: Full pipeline...")
    try:
        scored = []
        weights = config.get("weights", {})
        for cid, sem_score in list(hybrid_fused)[:200]:
            if cid in cand_lookup:
                attr = compute_attribute_score(cand_lookup[cid], jd, attr_weights)
                beh = compute_behavioral_score(cand_lookup[cid], config)
                norm_sem = sem_score / max_sem if max_sem > 0 else 0
                final = compute_final_score(norm_sem, attr["score"], beh["score"], weights)
                scored.append({"candidate_id": cid, "final_score": final})
        
        scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
        full_ranked = [s["candidate_id"] for s in scored[:100]]
        results["full_pipeline"] = evaluate_ranking(full_ranked, eval_set)
        results["full_pipeline"]["method"] = "Full pipeline (hybrid + attr + behavioral)"
    except Exception as e:
        results["full_pipeline"] = {"error": str(e), "method": "Full pipeline"}
    
    return results


def format_ablation_table(results: dict) -> str:
    """Format ablation results as a readable table."""
    header = f"{'Method':<40} {'Composite':>10} {'NDCG@10':>8} {'NDCG@50':>8} {'MAP':>8} {'P@10':>8}"
    separator = "-" * len(header)
    
    lines = [header, separator]
    
    for key in ["keyword_only", "embedding_only", "hybrid_no_behavioral", "full_pipeline"]:
        r = results.get(key, {})
        if "error" in r:
            lines.append(f"{r.get('method', key):<40} {'ERROR':>10}")
        else:
            lines.append(
                f"{r.get('method', key):<40} "
                f"{r.get('composite', 0):>10.4f} "
                f"{r.get('ndcg_10', 0):>8.4f} "
                f"{r.get('ndcg_50', 0):>8.4f} "
                f"{r.get('map', 0):>8.4f} "
                f"{r.get('p_10', 0):>8.4f}"
            )
    
    return "\n".join(lines)


def save_ablation_results(results: dict, filepath: str):
    """Save ablation results to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    print("Ablation study module ready.")
    print("Configurations: keyword-only, embedding-only, hybrid-no-behavioral, full-pipeline")
