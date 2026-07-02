"""
Honeypot candidate detector.
Identifies candidates with subtly impossible profiles that are traps
in the dataset. >10% honeypot rate in top-100 = disqualification.

Detection heuristics based on challenge documentation:
- Experience impossibilities (stated years vs career history sum)
- Skill impossibilities (expert proficiency with 0 months usage)
- Title/description mismatches
- Anomalous signal patterns
"""

import re
from datetime import datetime, date
from typing import Optional


def detect_honeypot(candidate: dict, config: dict = None) -> dict:
    """
    Analyze a candidate for honeypot indicators.
    
    Returns:
        dict with keys:
            - is_honeypot: bool
            - confidence: float (0-1)
            - reasons: list of strings explaining why
    """
    reasons = []
    flags = 0
    total_checks = 10  # total number of checks performed
    
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    education = candidate.get("education", [])
    
    # -------------------------------------------------------------------------
    # Check 1: Experience impossibility
    # Stated years_of_experience vs sum of career_history durations
    # -------------------------------------------------------------------------
    stated_exp = profile.get("years_of_experience", 0)
    career_months = sum(r.get("duration_months", 0) for r in career)
    career_years = career_months / 12.0
    
    # Allow some overlap (concurrent roles) but flag large gaps
    if stated_exp > 0 and career_years > 0:
        # If stated experience is MORE than career history by a large margin
        if stated_exp > career_years + 5:
            reasons.append(
                f"Experience gap: stated {stated_exp:.1f} yrs but career history sums to {career_years:.1f} yrs"
            )
            flags += 2
        # If career history is impossibly longer than stated
        if career_years > stated_exp * 2 + 3:
            reasons.append(
                f"Career history ({career_years:.1f} yrs) vastly exceeds stated experience ({stated_exp:.1f} yrs)"
            )
            flags += 1
    
    # -------------------------------------------------------------------------
    # Check 2: Expert skills with zero or very low duration
    # "Expert" proficiency in a skill with 0 months of usage is impossible
    # -------------------------------------------------------------------------
    expert_zero_count = 0
    for skill in skills:
        proficiency = skill.get("proficiency", "")
        duration = skill.get("duration_months", 0)
        if proficiency == "expert" and duration <= 3:
            expert_zero_count += 1
    
    if expert_zero_count >= 2:
        reasons.append(
            f"{expert_zero_count} skills listed as 'expert' with ≤3 months usage"
        )
        flags += 2
    elif expert_zero_count == 1:
        flags += 1
    
    # -------------------------------------------------------------------------
    # Check 3: Too many expert skills (statistical anomaly)
    # Having 8+ expert skills is extremely rare in real profiles
    # -------------------------------------------------------------------------
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count >= 8:
        reasons.append(f"{expert_count} skills at expert level (statistically unlikely)")
        flags += 1
    
    # -------------------------------------------------------------------------
    # Check 4: Title-description mismatch
    # E.g., title says "Marketing Manager" but description talks about ML engineering
    # -------------------------------------------------------------------------
    for role in career:
        title = (role.get("title", "") or "").lower()
        desc = (role.get("description", "") or "").lower()
        
        # Check for severe mismatches
        non_tech_titles = ["marketing", "accountant", "hr manager", "sales", 
                          "content writer", "graphic designer", "civil engineer",
                          "mechanical engineer", "operations manager", "customer support"]
        tech_keywords = ["machine learning", "deep learning", "neural network", 
                        "embeddings", "model training", "data pipeline", "ml model",
                        "fine-tuning", "llm", "transformer", "nlp"]
        
        title_is_non_tech = any(t in title for t in non_tech_titles)
        desc_has_tech = sum(1 for kw in tech_keywords if kw in desc)
        
        if title_is_non_tech and desc_has_tech >= 3:
            reasons.append(
                f"Title '{role.get('title')}' contradicts technical description"
            )
            flags += 1
    
    # -------------------------------------------------------------------------
    # Check 5: Impossible company tenure
    # E.g., 8 years at a company that was founded 3 years ago
    # (We check for durations that seem unreasonably long relative to dates)
    # -------------------------------------------------------------------------
    for role in career:
        start_date_str = role.get("start_date")
        end_date_str = role.get("end_date")
        duration_months = role.get("duration_months", 0)
        
        if start_date_str and duration_months > 0:
            try:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
                if end_date_str:
                    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
                else:
                    end_date = date(2026, 6, 15)  # reference date
                
                actual_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                
                # If stated duration is much more than actual date range
                if duration_months > actual_months + 12:
                    reasons.append(
                        f"Duration mismatch at {role.get('company', '?')}: "
                        f"stated {duration_months} months but dates span {actual_months} months"
                    )
                    flags += 2
            except (ValueError, TypeError):
                pass
    
    # -------------------------------------------------------------------------
    # Check 6: Skill assessment scores inconsistent with proficiency
    # High assessment scores in skills not listed, or very low for "expert" skills
    # -------------------------------------------------------------------------
    assessment_scores = signals.get("skill_assessment_scores", {})
    skill_names = {s.get("name", "").lower() for s in skills}
    
    for assessed_skill, score in assessment_scores.items():
        matching_skills = [s for s in skills if s.get("name", "").lower() == assessed_skill.lower()]
        if matching_skills:
            skill = matching_skills[0]
            if skill.get("proficiency") == "expert" and score < 30:
                reasons.append(
                    f"Expert in '{assessed_skill}' but assessment score only {score}"
                )
                flags += 1
    
    # -------------------------------------------------------------------------
    # Check 7: Impossible date sequences
    # Career entries with overlapping or impossible date sequences
    # -------------------------------------------------------------------------
    sorted_career = sorted(career, key=lambda r: r.get("start_date", ""), reverse=True)
    for i in range(len(sorted_career) - 1):
        curr = sorted_career[i]
        prev = sorted_career[i + 1]
        curr_start = curr.get("start_date", "")
        prev_end = prev.get("end_date", "")
        
        if curr_start and prev_end and curr.get("is_current") and prev.get("is_current"):
            reasons.append("Multiple concurrent 'is_current' roles")
            flags += 1
            break
    
    # -------------------------------------------------------------------------
    # Check 8: Endorsement anomalies
    # Very high endorsements on skills with zero duration
    # -------------------------------------------------------------------------
    for skill in skills:
        endorsements = skill.get("endorsements", 0)
        duration = skill.get("duration_months", 0)
        if endorsements > 30 and duration == 0:
            reasons.append(
                f"High endorsements ({endorsements}) for '{skill.get('name')}' with 0 months duration"
            )
            flags += 1
    
    # -------------------------------------------------------------------------
    # Check 9: Profile completeness vs actual content mismatch
    # Very high completeness score but missing key fields
    # -------------------------------------------------------------------------
    completeness = signals.get("profile_completeness_score", 0)
    has_summary = bool(profile.get("summary", "").strip())
    has_skills = len(skills) > 0
    has_education = len(education) > 0
    
    if completeness > 90 and (not has_summary or not has_skills or not has_education):
        reasons.append(
            f"Profile completeness {completeness}% but missing key fields"
        )
        flags += 1
    
    # -------------------------------------------------------------------------
    # Check 10: Education date anomalies
    # Graduated before plausible age, or education dates impossible
    # -------------------------------------------------------------------------
    for edu in education:
        start_year = edu.get("start_year", 0)
        end_year = edu.get("end_year", 0)
        if start_year > 0 and end_year > 0:
            duration = end_year - start_year
            if duration < 0:
                reasons.append(f"Education end year ({end_year}) before start year ({start_year})")
                flags += 2
            elif duration > 10:
                reasons.append(f"Education spanning {duration} years at {edu.get('institution', '?')}")
                flags += 1
    
    # Calculate confidence
    confidence = min(flags / 5.0, 1.0)  # Normalize to 0-1
    is_honeypot = flags >= 3 or (flags >= 2 and len(reasons) >= 2)
    
    return {
        "is_honeypot": is_honeypot,
        "confidence": confidence,
        "flags": flags,
        "reasons": reasons
    }


def filter_honeypots(candidates: list[dict], config: dict = None) -> tuple[list[dict], list[dict]]:
    """
    Separate candidates into clean and honeypot lists.
    
    Returns:
        (clean_candidates, honeypot_candidates) tuple
    """
    clean = []
    honeypots = []
    
    for candidate in candidates:
        result = detect_honeypot(candidate, config)
        candidate["_honeypot_check"] = result
        
        if result["is_honeypot"]:
            honeypots.append(candidate)
        else:
            clean.append(candidate)
    
    return clean, honeypots


if __name__ == "__main__":
    from loader import load_all_candidates, load_config
    
    config = load_config()
    candidates = load_all_candidates(config["paths"]["sample_candidates"])
    clean, honeypots = filter_honeypots(candidates)
    
    print(f"Total: {len(candidates)}, Clean: {len(clean)}, Honeypots: {len(honeypots)}")
    for hp in honeypots:
        check = hp["_honeypot_check"]
        print(f"\n  {hp['candidate_id']} ({hp['profile']['headline']})")
        for reason in check["reasons"]:
            print(f"    - {reason}")
