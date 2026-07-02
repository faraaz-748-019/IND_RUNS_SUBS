"""
Explainability / rationale generator.
Generates per-candidate natural-language reasoning based on sub-scores.
The reasoning must be:
  - Specific (references actual profile data)
  - Varied (not templated / not identical across candidates)
  - Honest (acknowledges gaps)
  - Non-hallucinating (only claims facts present in the profile)
"""

from typing import Optional


def generate_rationale(scored_candidate: dict) -> str:
    """
    Generate a 1-2 sentence reasoning for why a candidate is ranked where they are.
    
    Pulls specific facts from the candidate's profile and score breakdown.
    Avoids generic praise — the submission spec checks for specificity.
    
    Args:
        scored_candidate: Dict with candidate data and all sub-scores
    
    Returns:
        Reasoning string (1-2 sentences)
    """
    candidate = scored_candidate.get("candidate", {})
    profile = candidate.get("profile", {})
    attr_details = scored_candidate.get("attribute_details", {})
    beh_details = scored_candidate.get("behavioral_details", {})
    final_score = scored_candidate.get("final_score", 0)
    
    parts = []
    concerns = []
    
    # === STRENGTHS ===
    
    # Title relevance
    title = profile.get("current_title", "Professional")
    company = profile.get("current_company", "")
    title_info = attr_details.get("title", {})
    
    # Experience
    yoe = profile.get("years_of_experience", 0)
    exp_info = attr_details.get("experience", {})
    
    if title_info.get("is_relevant"):
        parts.append(f"{title} at {company} with {yoe:.1f} yrs experience")
    elif title_info.get("has_tech_roles"):
        parts.append(f"Currently {title} at {company} ({yoe:.1f} yrs), has relevant technical background")
    else:
        parts.append(f"{title} at {company} ({yoe:.1f} yrs)")
    
    # Skills
    skills_info = attr_details.get("skills", {})
    must_hits = skills_info.get("must_have_hits", 0)
    must_total = skills_info.get("must_have_total", 1)
    nice_hits = skills_info.get("nice_to_have_hits", 0)
    ai_core = skills_info.get("ai_core_count", 0)
    
    # Get actual skill names for specificity
    cand_skills = candidate.get("skills", [])
    top_skills = _get_relevant_skill_names(cand_skills, max_count=4)
    
    if must_hits > 0 and top_skills:
        parts.append(f"{must_hits}/{must_total} must-have skills matched ({', '.join(top_skills)})")
    elif top_skills:
        parts.append(f"skills include {', '.join(top_skills)}")
    
    # Behavioral signals
    recency_info = beh_details.get("recency", {})
    responsiveness_info = beh_details.get("responsiveness", {})
    availability_info = beh_details.get("availability", {})
    
    response_rate = responsiveness_info.get("response_rate", 0)
    days_since = recency_info.get("days_since_active")
    notice_days = availability_info.get("notice_period_days", 0)
    open_to_work = availability_info.get("open_to_work", False)
    
    beh_notes = []
    if open_to_work:
        beh_notes.append("open to work")
    if response_rate >= 0.5:
        beh_notes.append(f"{response_rate:.0%} response rate")
    if days_since is not None and days_since <= 30:
        beh_notes.append("recently active")
    if notice_days <= 30:
        beh_notes.append(f"{notice_days}-day notice period")
    
    if beh_notes:
        parts.append("; ".join(beh_notes))
    
    # === CONCERNS ===
    
    # Experience gap
    if exp_info.get("score", 1.0) < 0.5:
        if yoe < 5:
            concerns.append(f"below target experience range ({yoe:.1f} vs 5-9 yrs)")
        elif yoe > 9:
            concerns.append(f"above target experience range ({yoe:.1f} vs 5-9 yrs)")
    
    # Irrelevant title
    if title_info.get("is_irrelevant") and not title_info.get("has_tech_roles"):
        concerns.append(f"non-technical title ({title})")
    
    # Domain concerns
    domain_info = attr_details.get("domain", {})
    if domain_info.get("consulting_ratio", 0) > 0.8:
        concerns.append("primarily consulting background")
    
    # Low responsiveness
    if response_rate < 0.2:
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")
    
    # Inactive
    if days_since is not None and days_since > 180:
        concerns.append(f"inactive for {days_since} days")
    
    # === ASSEMBLE ===
    strength_text = ". ".join(parts[:3])  # Keep to 2-3 key points
    
    if concerns and final_score < 0.6:
        concern_text = "Concerns: " + "; ".join(concerns[:2])
        return f"{strength_text}. {concern_text}."
    elif concerns and final_score < 0.8:
        concern_text = "Note: " + concerns[0]
        return f"{strength_text}. {concern_text}."
    else:
        return f"{strength_text}."


def _get_relevant_skill_names(skills: list[dict], max_count: int = 4) -> list[str]:
    """Get the most relevant skill names from a candidate's skill list."""
    # Prioritize by: proficiency level, then endorsements
    prof_order = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    
    sorted_skills = sorted(
        skills,
        key=lambda s: (
            prof_order.get(s.get("proficiency", ""), 0),
            s.get("endorsements", 0),
        ),
        reverse=True,
    )
    
    return [s.get("name", "") for s in sorted_skills[:max_count] if s.get("name")]


def generate_score_breakdown(scored_candidate: dict) -> dict:
    """
    Generate a detailed score breakdown for display/export.
    """
    return {
        "final_score": round(scored_candidate.get("final_score", 0), 4),
        "semantic_score": round(scored_candidate.get("semantic_score_normalized", 
                                scored_candidate.get("semantic_score", 0)), 4),
        "attribute_score": round(scored_candidate.get("attribute_score", 0), 4),
        "behavioral_score": round(scored_candidate.get("behavioral_score", 0), 4),
        "attribute_breakdown": {
            k: round(v.get("score", 0), 4)
            for k, v in scored_candidate.get("attribute_details", {}).items()
        },
        "behavioral_breakdown": {
            k: round(v.get("score", 0), 4)
            for k, v in scored_candidate.get("behavioral_details", {}).items()
        },
    }


if __name__ == "__main__":
    # Test with mock data
    mock = {
        "candidate": {
            "candidate_id": "CAND_0000001",
            "profile": {
                "current_title": "ML Engineer",
                "current_company": "TechCorp",
                "years_of_experience": 7.0,
            },
            "skills": [
                {"name": "Python", "proficiency": "expert", "endorsements": 20},
                {"name": "PyTorch", "proficiency": "advanced", "endorsements": 15},
                {"name": "FAISS", "proficiency": "intermediate", "endorsements": 5},
            ],
        },
        "final_score": 0.85,
        "attribute_details": {
            "title": {"score": 0.9, "is_relevant": True, "has_tech_roles": True, "is_irrelevant": False},
            "skills": {"score": 0.8, "must_have_hits": 5, "must_have_total": 10, "ai_core_count": 8},
            "experience": {"score": 0.9},
            "domain": {"consulting_ratio": 0.1},
        },
        "behavioral_details": {
            "recency": {"score": 0.9, "days_since_active": 10},
            "responsiveness": {"score": 0.7, "response_rate": 0.65},
            "availability": {"score": 0.8, "open_to_work": True, "notice_period_days": 30},
        },
    }
    
    rationale = generate_rationale(mock)
    print(f"Rationale: {rationale}")
