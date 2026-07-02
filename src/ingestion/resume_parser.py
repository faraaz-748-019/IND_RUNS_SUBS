"""
Resume / unstructured text parser.
Demonstrates real-world robustness by extracting structured data
from messy, unstructured resume text.
Uses regex + heuristics — no paid APIs.
"""

import re
from typing import Optional


# Common skill keywords for extraction
SKILL_PATTERNS = [
    # Programming languages
    r'\b(Python|Java|JavaScript|TypeScript|Go|Rust|C\+\+|C#|Ruby|Scala|Kotlin|R|SQL)\b',
    # ML/AI frameworks  
    r'\b(PyTorch|TensorFlow|Keras|scikit-learn|sklearn|Hugging\s*Face|transformers|ONNX)\b',
    # Data/ML concepts
    r'\b(machine\s+learning|deep\s+learning|NLP|natural\s+language\s+processing|'
    r'computer\s+vision|reinforcement\s+learning|neural\s+networks?|embeddings?|'
    r'retrieval|ranking|recommendation|search|information\s+retrieval)\b',
    # Tools
    r'\b(Docker|Kubernetes|AWS|GCP|Azure|Spark|Airflow|Kafka|Redis|'
    r'PostgreSQL|MongoDB|Elasticsearch|FAISS|Pinecone|Weaviate)\b',
    # ML-specific
    r'\b(BERT|GPT|LLM|RAG|fine-?tuning|LoRA|PEFT|XGBoost|LightGBM|'
    r'sentence-?transformers?|NDCG|MRR|MAP|A/B\s+test)\b',
]

# Degree patterns
DEGREE_PATTERNS = [
    r'\b(Ph\.?D|Doctor(?:ate)?)\b',
    r'\b(M\.?(?:Tech|S|Sc|E|A|B\.?A)|Master(?:s)?(?:\s+of)?)\b',
    r'\b(B\.?(?:Tech|S|Sc|E|A)|Bachelor(?:s)?(?:\s+of)?)\b',
]

# Experience extraction
EXPERIENCE_PATTERN = r'(\d+\.?\d*)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)?'

# Email and phone
EMAIL_PATTERN = r'[\w.-]+@[\w.-]+\.\w+'
PHONE_PATTERN = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}'


def parse_resume_text(text: str) -> dict:
    """
    Extract structured data from unstructured resume text.
    
    Args:
        text: Raw resume text (from PDF, DOCX, or plain text)
    
    Returns:
        dict with extracted fields:
            - skills: list of skill strings
            - experience_years: float or None
            - education: list of degree strings
            - emails: list
            - phones: list
            - summary: first paragraph
    """
    result = {
        "skills": [],
        "experience_years": None,
        "education": [],
        "emails": [],
        "phones": [],
        "summary": "",
    }
    
    if not text or not text.strip():
        return result
    
    text_clean = text.strip()
    
    # Extract skills
    found_skills = set()
    for pattern in SKILL_PATTERNS:
        matches = re.findall(pattern, text_clean, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            found_skills.add(match.strip())
    result["skills"] = sorted(found_skills)
    
    # Extract experience years
    exp_matches = re.findall(EXPERIENCE_PATTERN, text_clean, re.IGNORECASE)
    if exp_matches:
        # Take the largest mentioned experience
        years = [float(y) for y in exp_matches]
        result["experience_years"] = max(years)
    
    # Extract education
    for pattern in DEGREE_PATTERNS:
        matches = re.findall(pattern, text_clean, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            result["education"].append(match.strip())
    result["education"] = list(set(result["education"]))
    
    # Extract contact info
    result["emails"] = re.findall(EMAIL_PATTERN, text_clean)
    result["phones"] = re.findall(PHONE_PATTERN, text_clean)
    
    # Extract summary (first non-empty paragraph)
    paragraphs = [p.strip() for p in text_clean.split("\n\n") if p.strip()]
    if paragraphs:
        # Skip if it looks like a name/header (short, all caps, etc.)
        for para in paragraphs:
            if len(para) > 50 and not para.isupper():
                result["summary"] = para
                break
    
    return result


def parse_resume_file(filepath: str) -> dict:
    """
    Parse a resume file (PDF, DOCX, or TXT) into structured data.
    
    Args:
        filepath: Path to resume file
    
    Returns:
        Extracted resume data dict
    """
    import os
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    elif ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            raise ImportError("pdfplumber required for PDF parsing. Install: pip install pdfplumber")
    elif ext in (".docx", ".doc"):
        try:
            from docx import Document
            doc = Document(filepath)
            text = "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            raise ImportError("python-docx required for DOCX parsing. Install: pip install python-docx")
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return parse_resume_text(text)


if __name__ == "__main__":
    # Test with sample resume text
    sample_resume = """
    John Doe
    Senior ML Engineer
    john.doe@email.com | +91-9876543210
    
    Experienced machine learning engineer with 7+ years of experience building 
    production ML systems. Specializing in NLP, information retrieval, and 
    recommendation systems.
    
    Skills: Python, PyTorch, TensorFlow, FAISS, Elasticsearch, Docker, AWS, 
    scikit-learn, BERT, sentence-transformers, RAG
    
    Education:
    M.Tech in Computer Science from IIT Delhi (2015-2017)
    B.Tech in Computer Science from NIT Trichy (2011-2015)
    
    Experience:
    Senior ML Engineer at SearchCo (2020-Present)
    - Built hybrid search system using BM25 + dense retrieval
    - Improved NDCG@10 by 15% through cross-encoder reranking
    
    ML Engineer at DataCorp (2017-2020)
    - Developed recommendation engine serving 1M+ users
    - Implemented A/B testing framework for model evaluation
    """
    
    result = parse_resume_text(sample_resume)
    print(f"Skills: {result['skills']}")
    print(f"Experience: {result['experience_years']} years")
    print(f"Education: {result['education']}")
    print(f"Emails: {result['emails']}")
