"""
Data loader for candidate profiles and job descriptions.
Supports streaming JSONL (memory-efficient for large files) and JSON formats.
Config-driven schema mapping allows swapping data sources without code changes.
"""

import json
import os
from pathlib import Path
from typing import Generator, Optional


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the master configuration file."""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stream_candidates(filepath: str, max_candidates: Optional[int] = None) -> Generator[dict, None, None]:
    """
    Stream candidates from JSONL file line-by-line.
    Memory-efficient: never loads all 100K candidates into RAM at once.
    
    Args:
        filepath: Path to .jsonl or .json file
        max_candidates: Optional limit on number of candidates to load
    
    Yields:
        dict: Individual candidate profile
    """
    filepath = Path(filepath)
    count = 0
    
    if filepath.suffix == ".jsonl":
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
                count += 1
                if max_candidates and count >= max_candidates:
                    return
    elif filepath.suffix == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for candidate in data:
                yield candidate
                count += 1
                if max_candidates and count >= max_candidates:
                    return
        else:
            yield data
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}. Use .jsonl or .json")


def load_all_candidates(filepath: str, max_candidates: Optional[int] = None) -> list[dict]:
    """
    Load all candidates into memory. Use for smaller datasets or when
    random access is needed (e.g., during scoring).
    
    Args:
        filepath: Path to .jsonl or .json file
        max_candidates: Optional limit
    
    Returns:
        List of candidate dicts
    """
    return list(stream_candidates(filepath, max_candidates))


def build_candidate_text(candidate: dict, include_signals: bool = False) -> str:
    """
    Build a composite text representation of a candidate for embedding/BM25.
    Concatenates the most informative text fields.
    
    Args:
        candidate: Candidate profile dict
        include_signals: Whether to include behavioral signals in text
    
    Returns:
        Concatenated text string
    """
    parts = []
    profile = candidate.get("profile", {})
    
    # Headline and summary are the richest text signals
    if profile.get("headline"):
        parts.append(profile["headline"])
    if profile.get("summary"):
        parts.append(profile["summary"])
    
    # Current role context
    if profile.get("current_title"):
        parts.append(f"Current role: {profile['current_title']}")
    if profile.get("current_company"):
        parts.append(f"at {profile['current_company']}")
    if profile.get("current_industry"):
        parts.append(f"Industry: {profile['current_industry']}")
    
    # Career history descriptions are gold for semantic matching
    for role in candidate.get("career_history", []):
        role_text = f"{role.get('title', '')} at {role.get('company', '')}"
        if role.get("description"):
            role_text += f": {role['description']}"
        parts.append(role_text)
    
    # Skills as a comma-separated list
    skills = candidate.get("skills", [])
    if skills:
        skill_names = [s.get("name", "") for s in skills if s.get("name")]
        parts.append(f"Skills: {', '.join(skill_names)}")
    
    # Education
    for edu in candidate.get("education", []):
        edu_text = f"{edu.get('degree', '')} in {edu.get('field_of_study', '')} from {edu.get('institution', '')}"
        parts.append(edu_text)
    
    # Certifications
    for cert in candidate.get("certifications", []):
        parts.append(f"Certification: {cert.get('name', '')} by {cert.get('issuer', '')}")
    
    return " . ".join(parts)


def load_job_description(filepath: str) -> str:
    """Load job description from a text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def apply_schema_mapping(candidate: dict, mapping: dict) -> dict:
    """
    Apply schema mapping to rename fields according to config.
    Allows swapping in different data sources by mapping column names.
    
    Args:
        candidate: Raw candidate dict
        mapping: Dict of {standard_name: source_name}
    
    Returns:
        Remapped candidate dict
    """
    result = {}
    for standard_key, source_key in mapping.items():
        if source_key in candidate:
            result[standard_key] = candidate[source_key]
    return result


if __name__ == "__main__":
    # Quick test with sample data
    config = load_config()
    sample_path = config["paths"]["sample_candidates"]
    
    candidates = load_all_candidates(sample_path, max_candidates=3)
    print(f"Loaded {len(candidates)} sample candidates")
    
    for c in candidates:
        text = build_candidate_text(c)
        print(f"\n{c['candidate_id']}: {c['profile']['headline']}")
        print(f"  Text length: {len(text)} chars")
