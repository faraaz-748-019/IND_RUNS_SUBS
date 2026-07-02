"""
Structured attribute scorer.
Scores candidates on hard-match attributes against JD requirements.
This is the single most important scorer for this specific JD because
the challenge explicitly rewards systems that distinguish real AI engineers
from keyword-stuffed non-technical profiles.

Sub-scores:
  - skills_match:      Overlap with must-have / nice-to-have skills
  - experience_fit:    Years of experience vs JD range
  - title_relevance:   Current/past title relevance
  - domain_fit:        Industry / product-company vs consulting
  - career_trajectory: Career path quality signals
  - location_fit:      Location / relocation alignment
  - education_fit:     Degree / field relevance
"""

import re
import math
from typing import Optional

# ---------------------------------------------------------------------------
# Skill synonyms — map common aliases to canonical forms so we don't miss
# matches caused by naming differences between JD and candidate profiles.
# ---------------------------------------------------------------------------
SKILL_SYNONYMS = {
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "ir": "information retrieval",
    "cv": "computer vision",
    "llm": "large language model",
    "llms": "large language model",
    "pytorch": "torch",
    "tf": "tensorflow",
    "sklearn": "scikit-learn",
    "sci-kit learn": "scikit-learn",
    "k8s": "kubernetes",
    "es": "elasticsearch",
    "genai": "generative ai",
    "gen ai": "generative ai",
    "sbert": "sentence-transformers",
    "sentence transformers": "sentence-transformers",
    "vector db": "vector database",
    "vectordb": "vector database",
    "ann": "approximate nearest neighbor",
    "rec sys": "recommendation",
    "recsys": "recommendation",
    "xgb": "xgboost",
    "lgbm": "lightgbm",
}

# Skills that are strongly AI/ML-core (for counting true AI depth)
AI_CORE_SKILLS = {
    "machine learning", "deep learning", "nlp", "natural language processing",
    "information retrieval", "embeddings", "transformers", "pytorch", "torch",
    "tensorflow", "keras", "scikit-learn", "faiss", "sentence-transformers",
    "bert", "gpt", "llm", "large language model", "fine-tuning", "lora",
    "qlora", "peft", "rag", "retrieval", "ranking", "recommendation",
    "search", "vector database", "neural network", "xgboost", "lightgbm",
    "reinforcement learning", "generative ai", "computer vision",
    "model training", "model deployment", "mlops", "data science",
    "feature engineering", "model evaluation", "a/b testing",
    "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch",
    "ndcg", "mrr", "map", "spark", "airflow", "python",
    "hugging face", "huggingface", "wandb", "weights & biases",
    "bentoml", "mlflow", "kubeflow",
}


def _normalize_skill(name: str) -> str:
    """Normalize a skill name for comparison."""
    n = name.lower().strip()
    return SKILL_SYNONYMS.get(n, n)


def score_skills(candidate: dict, jd: dict) -> dict:
    """
    Score skill overlap between candidate and JD requirements.
    Incorporates:
      - Must-have vs nice-to-have differentiation
      - Proficiency and duration weighting (trust multiplier)
      - Endorsement-based trust signal (catches keyword stuffers)
    """
    candidate_skills = candidate.get("skills", [])
    must_have = jd.get("must_have_skills", set())
    nice_to_have = jd.get("nice_to_have_skills", set())

    # Build candidate skill lookup with metadata
    cand_skill_map = {}
    for s in candidate_skills:
        norm = _normalize_skill(s.get("name", ""))
        cand_skill_map[norm] = {
            "proficiency": s.get("proficiency", "beginner"),
            "duration_months": s.get("duration_months", 0),
            "endorsements": s.get("endorsements", 0),
        }

    # Proficiency weights
    prof_weight = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}

    # Must-have matches
    must_hits = 0
    must_weighted = 0.0
    for skill in must_have:
        ns = _normalize_skill(skill)
        # Check exact and substring matches
        match = _find_skill_match(ns, cand_skill_map)
        if match:
            meta = cand_skill_map[match]
            w = prof_weight.get(meta["proficiency"], 0.3)
            # Trust multiplier: penalize "expert" with zero real duration
            duration_trust = min(meta["duration_months"] / 12.0, 1.0) if meta["duration_months"] > 0 else 0.15
            endorsement_trust = min(meta["endorsements"] / 10.0, 1.0) if meta["endorsements"] > 0 else 0.3
            trust = 0.5 * duration_trust + 0.5 * endorsement_trust
            must_hits += 1
            must_weighted += w * trust

    # Nice-to-have matches
    nice_hits = 0
    nice_weighted = 0.0
    for skill in nice_to_have:
        ns = _normalize_skill(skill)
        match = _find_skill_match(ns, cand_skill_map)
        if match:
            meta = cand_skill_map[match]
            w = prof_weight.get(meta["proficiency"], 0.3)
            nice_hits += 1
            nice_weighted += w

    # Count AI-core skills the candidate actually has
    ai_core_count = 0
    for norm_name in cand_skill_map:
        if norm_name in AI_CORE_SKILLS:
            ai_core_count += 1
    # Also check career descriptions for AI keywords
    for role in candidate.get("career_history", []):
        desc = (role.get("description", "") or "").lower()
        for kw in ["machine learning", "deep learning", "nlp", "embeddings",
                    "retrieval", "ranking", "recommendation", "search engine",
                    "vector", "neural", "transformer", "fine-tun", "model training"]:
            if kw in desc:
                ai_core_count += 1
                break  # count once per role

    must_score = must_weighted / max(len(must_have), 1)
    nice_score = nice_weighted / max(len(nice_to_have), 1)

    # Combined: must-have is 70% of skill score, nice-to-have is 30%
    combined = 0.7 * must_score + 0.3 * nice_score

    return {
        "score": min(combined, 1.0),
        "must_have_hits": must_hits,
        "must_have_total": len(must_have),
        "nice_to_have_hits": nice_hits,
        "nice_to_have_total": len(nice_to_have),
        "ai_core_count": ai_core_count,
    }


def _find_skill_match(target: str, skill_map: dict) -> Optional[str]:
    """Find a matching skill in the candidate's skill map (fuzzy)."""
    if target in skill_map:
        return target
    # Substring matching for multi-word skills
    for cand_skill in skill_map:
        if target in cand_skill or cand_skill in target:
            return cand_skill
    return None


def score_experience(candidate: dict, jd: dict) -> dict:
    """
    Score experience fit. JD asks for 5-9 years.
    Perfect fit at 6-8, acceptable at 5-9, penalized outside.
    """
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    exp_min, exp_max = jd.get("experience_range", [5, 9])
    ideal_min, ideal_max = exp_min + 1, exp_max - 1  # sweet spot

    if ideal_min <= yoe <= ideal_max:
        score = 1.0
    elif exp_min <= yoe <= exp_max:
        score = 0.8
    elif yoe < exp_min:
        # Too junior: linear decay
        gap = exp_min - yoe
        score = max(0.0, 0.8 - gap * 0.15)
    else:
        # Too senior (not a hard disqualifier but less ideal)
        gap = yoe - exp_max
        score = max(0.1, 0.7 - gap * 0.05)

    return {"score": score, "years": yoe, "range": [exp_min, exp_max]}


def score_title(candidate: dict, jd: dict) -> dict:
    """
    Score title/role relevance.
    The JD explicitly says non-tech titles like "Marketing Manager" are traps.
    Career history titles matter more than just the current title.
    """
    relevant_titles = jd.get("relevant_titles", [])
    irrelevant_titles = jd.get("irrelevant_titles", [])
    current_title = (candidate.get("profile", {}).get("current_title", "") or "").lower()

    # Check current title against irrelevant list (hard penalty)
    is_irrelevant = any(irr in current_title for irr in irrelevant_titles)
    is_relevant = any(rel in current_title for rel in relevant_titles)

    # Check career history for relevant roles
    career = candidate.get("career_history", [])
    relevant_role_months = 0
    total_months = 0
    has_any_tech_role = False

    for role in career:
        title = (role.get("title", "") or "").lower()
        months = role.get("duration_months", 0)
        total_months += months
        if any(rel in title for rel in relevant_titles):
            relevant_role_months += months
            has_any_tech_role = True
        # Also check description for tech work
        desc = (role.get("description", "") or "").lower()
        tech_indicators = ["ml", "machine learning", "data pipeline", "model",
                          "embeddings", "retrieval", "search", "nlp", "neural",
                          "deep learning", "ai", "recommendation"]
        if sum(1 for t in tech_indicators if t in desc) >= 3:
            has_any_tech_role = True

    if is_irrelevant and not has_any_tech_role:
        score = 0.05  # near-zero for clearly irrelevant
    elif is_relevant:
        score = 1.0
    elif has_any_tech_role:
        ratio = relevant_role_months / max(total_months, 1)
        score = 0.5 + 0.4 * ratio
    else:
        score = 0.15

    return {
        "score": score,
        "current_title": current_title,
        "is_relevant": is_relevant,
        "is_irrelevant": is_irrelevant,
        "has_tech_roles": has_any_tech_role,
    }


def score_domain(candidate: dict, jd: dict) -> dict:
    """
    Score industry/domain relevance.
    Prefers product companies over consulting firms.
    The JD explicitly says consulting-only careers are NOT a fit.
    """
    consulting_firms = jd.get("consulting_firms", [])
    product_industries = jd.get("product_industries", [])
    career = candidate.get("career_history", [])
    current_industry = (candidate.get("profile", {}).get("current_industry", "") or "").lower()

    consulting_months = 0
    product_months = 0
    total_months = 0

    for role in career:
        company = (role.get("company", "") or "").lower()
        industry = (role.get("industry", "") or "").lower()
        months = role.get("duration_months", 0)
        total_months += months

        if any(cf in company for cf in consulting_firms):
            consulting_months += months
        if any(pi in industry for pi in product_industries) or \
           any(pi in (role.get("company_size", "") or "").lower() for pi in ["startup"]):
            product_months += months

    # All-consulting career = strong negative
    if total_months > 0:
        consulting_ratio = consulting_months / total_months
        product_ratio = product_months / total_months
    else:
        consulting_ratio = 0
        product_ratio = 0

    if consulting_ratio > 0.9:
        score = 0.1  # Consulting-only = bad fit per JD
    elif product_ratio > 0.5:
        score = 0.9
    elif product_ratio > 0.2:
        score = 0.6
    else:
        score = 0.35

    # Boost for relevant current industry
    domain_kws = jd.get("domain_keywords", set())
    if any(kw in current_industry for kw in domain_kws):
        score = min(score + 0.1, 1.0)

    return {
        "score": score,
        "consulting_ratio": consulting_ratio,
        "product_ratio": product_ratio,
    }


def score_career_trajectory(candidate: dict, jd: dict) -> dict:
    """
    Score career trajectory quality.
    Penalize: title-chasers (short stints), non-progressing careers.
    Reward: stable tenure, progressive responsibility, production ML work.
    """
    career = candidate.get("career_history", [])
    if not career:
        return {"score": 0.2, "avg_tenure_months": 0, "num_roles": 0}

    tenures = [r.get("duration_months", 0) for r in career]
    avg_tenure = sum(tenures) / len(tenures) if tenures else 0

    # Penalize very short average tenures (title-chasers, <18 months avg)
    if avg_tenure < 12:
        tenure_score = 0.2
    elif avg_tenure < 18:
        tenure_score = 0.5
    elif avg_tenure < 24:
        tenure_score = 0.7
    else:
        tenure_score = 1.0

    # Check for production ML/AI work in descriptions
    production_signals = 0
    for role in career:
        desc = (role.get("description", "") or "").lower()
        if any(kw in desc for kw in ["production", "deployed", "shipped",
                                      "users", "scale", "real-time", "serving",
                                      "pipeline", "a/b test", "metrics"]):
            production_signals += 1

    production_score = min(production_signals / 2.0, 1.0)

    score = 0.5 * tenure_score + 0.5 * production_score
    return {
        "score": score,
        "avg_tenure_months": avg_tenure,
        "num_roles": len(career),
        "production_signals": production_signals,
    }


def score_location(candidate: dict, jd: dict) -> dict:
    """Score location and relocation alignment."""
    location = (candidate.get("profile", {}).get("location", "") or "").lower()
    country = (candidate.get("profile", {}).get("country", "") or "").lower()
    willing_to_relocate = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)
    preferred_locations = [l.lower() for l in jd.get("location_preferred", [])]

    is_in_preferred = any(loc in location for loc in preferred_locations)
    is_india = "india" in country

    if is_in_preferred:
        score = 1.0
    elif is_india:
        score = 0.7 if willing_to_relocate else 0.5
    elif willing_to_relocate:
        score = 0.4
    else:
        score = 0.2

    return {"score": score, "location": location, "country": country}


def score_education(candidate: dict, jd: dict) -> dict:
    """Score education relevance."""
    education = candidate.get("education", [])
    if not education:
        return {"score": 0.3, "highest_degree": "none", "field_relevant": False}

    relevant_fields = {"computer science", "machine learning", "artificial intelligence",
                       "data science", "information technology", "software engineering",
                       "electrical engineering", "electronics", "mathematics",
                       "statistics", "computational", "cs"}
    degree_weight = {"ph.d": 1.0, "phd": 1.0, "m.tech": 0.9, "m.e.": 0.9,
                     "m.sc": 0.8, "m.s": 0.8, "mba": 0.5,
                     "b.tech": 0.7, "b.e.": 0.7, "b.sc": 0.6, "b.s": 0.6}
    tier_weight = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.5, "tier_4": 0.3, "unknown": 0.4}

    best_score = 0.0
    highest_degree = ""
    field_relevant = False

    for edu in education:
        field = (edu.get("field_of_study", "") or "").lower()
        degree = (edu.get("degree", "") or "").lower()
        tier = edu.get("tier", "unknown")

        is_relevant_field = any(rf in field for rf in relevant_fields)
        if is_relevant_field:
            field_relevant = True

        dw = max((v for k, v in degree_weight.items() if k in degree), default=0.4)
        tw = tier_weight.get(tier, 0.4)
        fw = 1.0 if is_relevant_field else 0.4

        edu_score = dw * 0.4 + tw * 0.3 + fw * 0.3
        if edu_score > best_score:
            best_score = edu_score
            highest_degree = degree

    return {"score": min(best_score, 1.0), "highest_degree": highest_degree, "field_relevant": field_relevant}


def compute_attribute_score(candidate: dict, jd: dict, weights: dict) -> dict:
    """
    Compute the full attribute score for a candidate.
    Returns individual sub-scores and the weighted combination.
    """
    skills = score_skills(candidate, jd)
    experience = score_experience(candidate, jd)
    title = score_title(candidate, jd)
    domain = score_domain(candidate, jd)
    trajectory = score_career_trajectory(candidate, jd)
    location = score_location(candidate, jd)
    education = score_education(candidate, jd)

    w = weights
    combined = (
        w.get("skills_match", 0.25) * skills["score"] +
        w.get("experience_fit", 0.20) * experience["score"] +
        w.get("title_relevance", 0.20) * title["score"] +
        w.get("domain_fit", 0.15) * domain["score"] +
        w.get("career_trajectory", 0.10) * trajectory["score"] +
        w.get("location_fit", 0.05) * location["score"] +
        w.get("education_fit", 0.05) * education["score"]
    )

    return {
        "score": combined,
        "sub_scores": {
            "skills": skills,
            "experience": experience,
            "title": title,
            "domain": domain,
            "career_trajectory": trajectory,
            "location": location,
            "education": education,
        }
    }


if __name__ == "__main__":
    import json, yaml, sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.ingestion.loader import load_config, load_all_candidates, load_job_description
    from src.ingestion.jd_parser import parse_jd

    config = load_config()
    jd_text = load_job_description(config["paths"]["job_description"])
    jd = parse_jd(jd_text, config)
    candidates = load_all_candidates(config["paths"]["sample_candidates"], max_candidates=5)

    for c in candidates:
        result = compute_attribute_score(c, jd, config["attribute_weights"])
        print(f"\n{c['candidate_id']} ({c['profile']['current_title']}): {result['score']:.3f}")
        for k, v in result["sub_scores"].items():
            print(f"  {k}: {v['score']:.3f}")
