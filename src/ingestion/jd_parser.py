"""
Job description parser.
Extracts structured requirements from JD text using rule-based NLP.
No paid LLM needed — pure regex + keyword extraction.
"""

import re
from typing import Optional


def parse_jd(jd_text: str, config: dict) -> dict:
    """
    Parse job description into structured requirements.
    Uses config-driven skill lists supplemented by text extraction.
    
    Args:
        jd_text: Raw job description text
        config: Master configuration dict
    
    Returns:
        Structured JD requirements dict
    """
    jd_config = config.get("jd", {})
    
    # Start with config-defined requirements (hand-tuned for this JD)
    requirements = {
        "title": jd_config.get("title", ""),
        "company": jd_config.get("company", ""),
        "experience_range": jd_config.get("experience_range", [5, 9]),
        "location_preferred": jd_config.get("location_preferred", []),
        "work_mode": jd_config.get("work_mode", ""),
        "notice_period_max": jd_config.get("notice_period_preferred_max", 30),
        "must_have_skills": set(s.lower() for s in jd_config.get("must_have_skills", [])),
        "nice_to_have_skills": set(s.lower() for s in jd_config.get("nice_to_have_skills", [])),
        "domain_keywords": set(s.lower() for s in jd_config.get("domain_keywords", [])),
        "relevant_titles": [t.lower() for t in jd_config.get("relevant_titles", [])],
        "irrelevant_titles": [t.lower() for t in jd_config.get("irrelevant_titles", [])],
        "product_industries": [i.lower() for i in jd_config.get("product_industries", [])],
        "consulting_firms": [f.lower() for f in jd_config.get("consulting_firms", [])],
    }
    
    # Supplement with text extraction from JD
    jd_lower = jd_text.lower()
    
    # Extract additional skill keywords from JD text
    # Match known tech terms that appear in the JD
    tech_patterns = [
        r'\b(python|java|go|rust|c\+\+|javascript|typescript)\b',
        r'\b(pytorch|tensorflow|keras|scikit-learn|sklearn)\b',
        r'\b(docker|kubernetes|k8s|aws|gcp|azure)\b',
        r'\b(sql|nosql|mongodb|postgresql|redis|kafka)\b',
        r'\b(bert|gpt|llm|transformer|attention|embedding)\b',
        r'\b(faiss|pinecone|weaviate|qdrant|milvus|elasticsearch|opensearch)\b',
        r'\b(rag|retrieval|ranking|recommendation|search|nlp|nlu)\b',
        r'\b(spark|airflow|dbt|snowflake|bigquery|databricks)\b',
        r'\b(mlops|ci/cd|git|linux|api|rest|grpc)\b',
    ]
    
    extracted_skills = set()
    for pattern in tech_patterns:
        matches = re.findall(pattern, jd_lower)
        extracted_skills.update(matches)
    
    requirements["extracted_skills"] = extracted_skills
    
    # Extract experience range from text if not in config
    exp_pattern = r'(\d+)\s*[-–]\s*(\d+)\s*years?'
    exp_matches = re.findall(exp_pattern, jd_text)
    if exp_matches and not requirements["experience_range"]:
        requirements["experience_range"] = [int(exp_matches[0][0]), int(exp_matches[0][1])]
    
    # Build the JD embedding text (for semantic matching)
    requirements["embedding_text"] = _build_jd_embedding_text(jd_text, requirements)
    
    return requirements


def _build_jd_embedding_text(jd_text: str, requirements: dict) -> str:
    """Build optimized text for JD embedding."""
    parts = [
        f"Senior AI Engineer role requiring {requirements['experience_range'][0]}-{requirements['experience_range'][1]} years experience.",
        "Must have: " + ", ".join(requirements["must_have_skills"]),
        "Nice to have: " + ", ".join(requirements["nice_to_have_skills"]),
        "Domain: " + ", ".join(requirements["domain_keywords"]),
        # Include key JD phrases for semantic matching
        "Production experience with embeddings-based retrieval systems.",
        "Vector databases and hybrid search infrastructure.",
        "Evaluation frameworks for ranking systems NDCG MRR MAP.",
        "NLP information retrieval search recommendation systems.",
        "Applied ML AI at product companies, not consulting.",
        "Strong Python, code quality, shipping to production.",
    ]
    return " ".join(parts)


if __name__ == "__main__":
    from loader import load_config, load_job_description
    
    config = load_config()
    jd_text = load_job_description(config["paths"]["job_description"])
    requirements = parse_jd(jd_text, config)
    
    print(f"Title: {requirements['title']}")
    print(f"Experience: {requirements['experience_range']}")
    print(f"Must-have skills ({len(requirements['must_have_skills'])}): {requirements['must_have_skills']}")
    print(f"Nice-to-have ({len(requirements['nice_to_have_skills'])}): {requirements['nice_to_have_skills']}")
    print(f"Irrelevant titles: {requirements['irrelevant_titles']}")
