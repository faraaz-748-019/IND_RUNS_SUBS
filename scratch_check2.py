import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

base = r'c:\Users\Acer\Desktop\IND R DAI\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge'

from docx import Document
for fname in ['job_description.docx', 'submission_spec.docx', 'redrob_signals_doc.docx']:
    fpath = os.path.join(base, fname)
    if os.path.exists(fpath):
        doc = Document(fpath)
        text = '\n'.join([p.text for p in doc.paragraphs])
        print(f"\n{'='*60}")
        print(f"FILE: {fname} ({len(text)} chars)")
        print(f"{'='*60}")
        print(text)
