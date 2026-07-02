"""
Bias masking utilities.
Strip/mask attributes that could introduce demographic bias before scoring.
Ensures ranking is driven by qualifications and signals, not identity proxies.
This is a responsible-AI feature documented in the README.
"""

import re
from typing import Optional
from copy import deepcopy


# Gender-coded terms to mask
GENDER_CODED_TERMS = [
    # Pronouns
    r'\bhe\b', r'\bhis\b', r'\bhim\b', r'\bhimself\b',
    r'\bshe\b', r'\bher\b', r'\bhers\b', r'\bherself\b',
    # Gendered titles
    r'\bmr\.?\b', r'\bmrs\.?\b', r'\bms\.?\b', r'\bmiss\b',
    # Gendered nouns
    r'\bbrotherhood\b', r'\bsisterhood\b',
    r'\bchairman\b', r'\bchairwoman\b',
]

# Age-indicator patterns
AGE_PATTERNS = [
    r'\b\d{2}\s*years?\s*old\b',
    r'\bage[d]?\s*\d{2}\b',
    r'\bborn\s+in\s+\d{4}\b',
    r'\bdate\s+of\s+birth\b',
    r'\bdob\b',
]


def mask_candidate_for_scoring(candidate: dict) -> dict:
    """
    Create a bias-masked copy of a candidate profile for scoring.
    Strips demographic signals while preserving qualification data.
    
    Masked fields:
    - anonymized_name (replaced with candidate_id)
    - Gender-coded pronouns in text fields
    - Age indicators
    
    Preserved fields (needed for scoring):
    - skills, experience, education, career_history
    - redrob_signals (behavioral data)
    - All qualification-relevant fields
    
    Args:
        candidate: Original candidate dict
    
    Returns:
        Bias-masked copy (original is not modified)
    """
    masked = deepcopy(candidate)
    
    # Replace name with anonymous identifier
    if "profile" in masked:
        masked["profile"]["anonymized_name"] = masked.get("candidate_id", "CANDIDATE")
    
    # Mask text fields
    text_fields_to_mask = []
    
    # Profile text fields
    if "profile" in masked:
        for field in ["headline", "summary"]:
            if field in masked["profile"] and masked["profile"][field]:
                masked["profile"][field] = _mask_text(masked["profile"][field])
    
    # Career history descriptions
    for role in masked.get("career_history", []):
        if "description" in role and role["description"]:
            role["description"] = _mask_text(role["description"])
    
    return masked


def _mask_text(text: str) -> str:
    """
    Remove gender-coded terms and age indicators from text.
    Replaces with neutral alternatives where possible.
    """
    if not text:
        return text
    
    masked = text
    
    # Mask gender-coded terms
    for pattern in GENDER_CODED_TERMS:
        masked = re.sub(pattern, "", masked, flags=re.IGNORECASE)
    
    # Mask age indicators
    for pattern in AGE_PATTERNS:
        masked = re.sub(pattern, "[MASKED]", masked, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    masked = re.sub(r'\s+', ' ', masked).strip()
    
    return masked


def audit_bias_masking(original: dict, masked: dict) -> dict:
    """
    Audit what was masked for transparency reporting.
    
    Returns:
        dict with masking audit information
    """
    changes = []
    
    # Check name masking
    orig_name = original.get("profile", {}).get("anonymized_name", "")
    masked_name = masked.get("profile", {}).get("anonymized_name", "")
    if orig_name != masked_name:
        changes.append(f"Name masked: '{orig_name}' → '{masked_name}'")
    
    # Check text field changes
    for field in ["headline", "summary"]:
        orig_text = original.get("profile", {}).get(field, "")
        masked_text = masked.get("profile", {}).get(field, "")
        if orig_text != masked_text:
            changes.append(f"Profile.{field}: text modified ({len(orig_text)} → {len(masked_text)} chars)")
    
    # Check career descriptions
    for i, (orig_role, masked_role) in enumerate(zip(
        original.get("career_history", []),
        masked.get("career_history", [])
    )):
        if orig_role.get("description") != masked_role.get("description"):
            changes.append(f"Career[{i}].description: text modified")
    
    return {
        "candidate_id": original.get("candidate_id"),
        "fields_modified": len(changes),
        "changes": changes
    }


if __name__ == "__main__":
    # Test with a sample candidate
    sample = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "John Smith",
            "headline": "He is a senior ML engineer",
            "summary": "Mr. Smith has built his career in machine learning. He specializes in NLP.",
        },
        "career_history": [
            {
                "description": "He led the ML team and built his own evaluation framework."
            }
        ]
    }
    
    masked = mask_candidate_for_scoring(sample)
    audit = audit_bias_masking(sample, masked)
    
    print("Original name:", sample["profile"]["anonymized_name"])
    print("Masked name:", masked["profile"]["anonymized_name"])
    print("\nOriginal headline:", sample["profile"]["headline"])
    print("Masked headline:", masked["profile"]["headline"])
    print("\nAudit:", audit)
