"""
Behavioral / activity signal scorer.
Scores candidates on Redrob platform signals: recency, responsiveness,
engagement, availability, assessments, social proof, and verification.

Uses exponential decay for recency so old activity counts less.
All signals normalized to 0-1 range.
"""

import math
from datetime import datetime, date
from typing import Optional


def _parse_date(date_str: str) -> Optional[date]:
    """Parse a date string safely."""
    if not date_str:
        return None
    try:
        return datetime.strptime(str(date_str), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _exponential_decay(days_ago: float, half_life: float = 90.0) -> float:
    """
    Exponential decay function.
    Returns 1.0 for days_ago=0, 0.5 for days_ago=half_life.
    """
    if days_ago <= 0:
        return 1.0
    return math.exp(-0.693 * days_ago / half_life)  # ln(2) ≈ 0.693


def score_recency(candidate: dict, config: dict) -> dict:
    """
    Score based on last_active_date with exponential decay.
    A candidate who hasn't logged in for 6 months is practically unavailable.
    """
    signals = candidate.get("redrob_signals", {})
    decay_config = config.get("decay", {})
    half_life = decay_config.get("half_life_days", 90)
    ref_date_str = decay_config.get("reference_date", "2026-06-15")
    ref_date = _parse_date(ref_date_str) or date(2026, 6, 15)

    last_active = _parse_date(signals.get("last_active_date"))
    if last_active is None:
        return {"score": 0.1, "days_since_active": None}

    days_ago = (ref_date - last_active).days
    if days_ago < 0:
        days_ago = 0

    score = _exponential_decay(days_ago, half_life)
    return {"score": score, "days_since_active": days_ago}


def score_responsiveness(candidate: dict, config: dict) -> dict:
    """
    Score recruiter responsiveness.
    Combines response rate and response time.
    """
    signals = candidate.get("redrob_signals", {})
    response_rate = signals.get("recruiter_response_rate", 0.0)
    response_time = signals.get("avg_response_time_hours", 999)

    # Response rate (0-1) — directly usable
    rate_score = response_rate

    # Response time — lower is better
    # < 24 hrs = excellent, < 72 hrs = good, > 168 hrs (1 week) = poor
    if response_time <= 12:
        time_score = 1.0
    elif response_time <= 24:
        time_score = 0.9
    elif response_time <= 48:
        time_score = 0.7
    elif response_time <= 72:
        time_score = 0.5
    elif response_time <= 168:
        time_score = 0.3
    else:
        time_score = 0.1

    score = 0.6 * rate_score + 0.4 * time_score
    return {
        "score": score,
        "response_rate": response_rate,
        "avg_response_time": response_time,
    }


def score_engagement(candidate: dict, config: dict) -> dict:
    """
    Score platform engagement: profile completeness, applications, views.
    """
    signals = candidate.get("redrob_signals", {})

    completeness = signals.get("profile_completeness_score", 0) / 100.0
    apps = signals.get("applications_submitted_30d", 0)
    views = signals.get("profile_views_received_30d", 0)
    search_appearances = signals.get("search_appearance_30d", 0)

    # Normalize apps (>5 is very active)
    apps_score = min(apps / 5.0, 1.0)
    # Normalize views (>20 is good visibility)
    views_score = min(views / 20.0, 1.0)
    # Search appearances (>100 is good)
    search_score = min(search_appearances / 100.0, 1.0)

    score = 0.35 * completeness + 0.25 * apps_score + 0.2 * views_score + 0.2 * search_score
    return {
        "score": score,
        "completeness": completeness * 100,
        "applications_30d": apps,
        "views_30d": views,
    }


def score_availability(candidate: dict, config: dict) -> dict:
    """
    Score availability signals.
    JD prefers: open_to_work=True, notice_period <= 30 days.
    """
    signals = candidate.get("redrob_signals", {})
    jd_config = config.get("jd", {})

    open_to_work = signals.get("open_to_work_flag", False)
    notice_days = signals.get("notice_period_days", 90)
    preferred_max = jd_config.get("notice_period_preferred_max", 30)

    # Open to work flag
    otw_score = 1.0 if open_to_work else 0.3

    # Notice period (lower is better for this JD)
    if notice_days <= preferred_max:
        notice_score = 1.0
    elif notice_days <= 60:
        notice_score = 0.6
    elif notice_days <= 90:
        notice_score = 0.4
    else:
        notice_score = 0.2

    # Work mode alignment
    work_mode = signals.get("preferred_work_mode", "")
    jd_work_mode = jd_config.get("work_mode", "hybrid")
    if work_mode == jd_work_mode or work_mode == "flexible":
        mode_score = 1.0
    elif work_mode in ("hybrid", "remote"):
        mode_score = 0.7
    else:
        mode_score = 0.5

    score = 0.4 * otw_score + 0.35 * notice_score + 0.25 * mode_score
    return {
        "score": score,
        "open_to_work": open_to_work,
        "notice_period_days": notice_days,
        "work_mode": work_mode,
    }


def score_assessments(candidate: dict, config: dict) -> dict:
    """
    Score platform skill assessments.
    Higher assessment scores in relevant skills = stronger candidate.
    """
    signals = candidate.get("redrob_signals", {})
    assessment_scores = signals.get("skill_assessment_scores", {})

    if not assessment_scores:
        return {"score": 0.3, "num_assessments": 0, "avg_score": 0}

    scores = list(assessment_scores.values())
    avg_score = sum(scores) / len(scores) if scores else 0
    num_assessments = len(scores)

    # Normalize: 0-100 scale → 0-1
    normalized = avg_score / 100.0
    # Bonus for taking multiple assessments
    volume_bonus = min(num_assessments / 5.0, 1.0) * 0.2

    score = min(normalized + volume_bonus, 1.0)
    return {
        "score": score,
        "num_assessments": num_assessments,
        "avg_score": avg_score,
    }


def score_social_proof(candidate: dict, config: dict) -> dict:
    """
    Score social proof signals: saved by recruiters, endorsements, GitHub.
    """
    signals = candidate.get("redrob_signals", {})

    saved = signals.get("saved_by_recruiters_30d", 0)
    endorsements = signals.get("endorsements_received", 0)
    connections = signals.get("connection_count", 0)
    github = signals.get("github_activity_score", -1)
    interview_rate = signals.get("interview_completion_rate", 0)
    offer_rate = signals.get("offer_acceptance_rate", -1)

    saved_score = min(saved / 10.0, 1.0)
    endorsement_score = min(endorsements / 30.0, 1.0)
    connection_score = min(connections / 300.0, 1.0)
    github_score = max(github, 0) / 100.0 if github >= 0 else 0.3
    interview_score = interview_rate
    offer_score = offer_rate if offer_rate >= 0 else 0.5

    score = (0.25 * saved_score + 0.2 * github_score + 0.15 * endorsement_score +
             0.15 * interview_score + 0.15 * offer_score + 0.1 * connection_score)
    return {
        "score": score,
        "saved_by_recruiters": saved,
        "github_score": github,
        "interview_completion": interview_rate,
    }


def score_verification(candidate: dict, config: dict) -> dict:
    """Score account verification status."""
    signals = candidate.get("redrob_signals", {})

    verified_email = signals.get("verified_email", False)
    verified_phone = signals.get("verified_phone", False)
    linkedin = signals.get("linkedin_connected", False)

    verified_count = sum([verified_email, verified_phone, linkedin])
    score = verified_count / 3.0

    return {
        "score": score,
        "email": verified_email,
        "phone": verified_phone,
        "linkedin": linkedin,
    }


def compute_behavioral_score(candidate: dict, config: dict) -> dict:
    """
    Compute the full behavioral score for a candidate.
    Returns individual sub-scores and weighted combination.
    """
    weights = config.get("behavioral_weights", {})

    recency = score_recency(candidate, config)
    responsiveness = score_responsiveness(candidate, config)
    engagement = score_engagement(candidate, config)
    availability = score_availability(candidate, config)
    assessment = score_assessments(candidate, config)
    social = score_social_proof(candidate, config)
    verification = score_verification(candidate, config)

    combined = (
        weights.get("recency", 0.25) * recency["score"] +
        weights.get("responsiveness", 0.20) * responsiveness["score"] +
        weights.get("engagement", 0.15) * engagement["score"] +
        weights.get("availability", 0.15) * availability["score"] +
        weights.get("assessment", 0.10) * assessment["score"] +
        weights.get("social_proof", 0.10) * social["score"] +
        weights.get("verification", 0.05) * verification["score"]
    )

    return {
        "score": combined,
        "sub_scores": {
            "recency": recency,
            "responsiveness": responsiveness,
            "engagement": engagement,
            "availability": availability,
            "assessment": assessment,
            "social_proof": social,
            "verification": verification,
        }
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.ingestion.loader import load_config, load_all_candidates

    config = load_config()
    candidates = load_all_candidates(config["paths"]["sample_candidates"], max_candidates=5)

    for c in candidates:
        result = compute_behavioral_score(c, config)
        print(f"\n{c['candidate_id']} ({c['profile']['current_title']}): {result['score']:.3f}")
        for k, v in result["sub_scores"].items():
            print(f"  {k}: {v['score']:.3f}")
