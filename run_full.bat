@echo off
echo Copying candidates.jsonl...
copy "c:\Users\Acer\Desktop\IND R DAI\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl" "c:\Users\Acer\Desktop\IND R DAI\data\candidates.jsonl"

echo Running pre-computation...
C:\Users\Acer\AppData\Local\Python\bin\python3.exe precompute.py --candidates data\candidates.jsonl

echo Running ranking engine...
C:\Users\Acer\AppData\Local\Python\bin\python3.exe rank.py --candidates data\candidates.jsonl --out output\submission.csv

echo Generating XLSX...
C:\Users\Acer\AppData\Local\Python\bin\python3.exe -c "import pandas as pd; pd.read_csv('output/submission.csv').to_excel('output/submission.xlsx', index=False)"

echo Validating output...
C:\Users\Acer\AppData\Local\Python\bin\python3.exe "c:\Users\Acer\Desktop\IND R DAI\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\validate_submission.py" "c:\Users\Acer\Desktop\IND R DAI\output\submission.csv"

echo Done!
