import json, os

base = r'c:\Users\Acer\Desktop\IND R DAI\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge'

# Sample candidates
with open(os.path.join(base, 'sample_candidates.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"Sample candidates count: {len(data)}")
print(f"First 3 IDs: {[c['candidate_id'] for c in data[:3]]}")

# Check JSONL file size
jsonl_path = os.path.join(base, 'candidates.jsonl')
size_mb = os.path.getsize(jsonl_path) / (1024*1024)
print(f"\ncandidates.jsonl size: {size_mb:.1f} MB")

# Count lines in JSONL
with open(jsonl_path, 'r', encoding='utf-8') as f:
    count = 0
    first_line = None
    for line in f:
        line = line.strip()
        if line:
            count += 1
            if first_line is None:
                first_line = line
print(f"candidates.jsonl line count: {count}")

# Parse first line
first = json.loads(first_line)
print(f"First JSONL candidate_id: {first['candidate_id']}")
print(f"Keys: {list(first.keys())}")

# Read docx files as text (try with python-docx or just report)
try:
    from docx import Document
    for fname in ['README.docx', 'job_description.docx', 'submission_spec.docx', 'redrob_signals_doc.docx']:
        fpath = os.path.join(base, fname)
        if os.path.exists(fpath):
            doc = Document(fpath)
            text = '\n'.join([p.text for p in doc.paragraphs])
            print(f"\n{'='*60}")
            print(f"FILE: {fname} ({len(text)} chars)")
            print(f"{'='*60}")
            print(text[:3000])
            print("... [truncated]" if len(text) > 3000 else "")
except ImportError:
    print("\npython-docx not installed, skipping docx reading")
