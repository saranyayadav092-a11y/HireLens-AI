# HireLens AI 🎯

**AI-assisted resume screening and job matching — built with Python and Streamlit.**

HireLens AI helps a recruiter compare multiple resumes against a job description, inspect matching skills, identify skills to verify, and export a shortlist for review.

## Features

- Bulk PDF and DOCX resume upload
- PDF text extraction with PyMuPDF
- DOCX text extraction with python-docx
- TF-IDF + cosine-similarity text relevance
- Configurable required-skill coverage
- Ranked shortlist with transparent skill comparisons
- Candidate detail view with extracted resume text
- CSV export
- Fictional sample profiles for hackathon demos
- Responsible-AI guidance and human-review reminders

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Deploy

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect GitHub and select this repository.
4. Choose `app.py` as the entry point.
5. Deploy.

## GitHub setup

Create an empty repository on GitHub, then from this folder run:

```bash
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "Build HireLens AI MVP"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/hirelens-ai.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

## How scoring works

When required skills are detected or supplied:

`score = (text relevance × configured weight) + (required skill coverage × remaining weight)`

The default is 60% text relevance and 40% required-skill coverage. Both are scaled to 0–100 before combination. If no required skills are provided, the app uses text relevance alone.

These weights are demo defaults, not scientifically validated measures of job performance. The score is a document-relevance indicator, not a probability of success or a candidate quality verdict.

## Current limitations

- The skill dictionary is intentionally small and must be expanded for each domain.
- TF-IDF does not fully understand meaning or transferable skills.
- Experience duration and qualifications are not reliably validated.
- Scanned/image-only PDFs may require OCR.
- Text extraction can miss information in complex resume layouts.
- The app is a hackathon prototype, not a validated hiring system.

## Responsible use

- Keep a human recruiter responsible for all hiring decisions.
- Do not automatically reject candidates based on scores.
- Do not score protected or irrelevant personal characteristics.
- Use fictional resumes for public demos.
- Do not commit resumes, candidate data, secrets, or exports to GitHub.
- Before real use, implement appropriate authentication, access controls, secure storage, deletion/retention policies, testing, and legal review.

## Suggested next upgrades

1. Semantic matching with sentence embeddings.
2. Structured extraction of experience, education, and projects with evidence.
3. OCR for scanned PDFs.
4. Recruiter-configurable job criteria and must-have requirements.
5. Evaluation on synthetic test cases and fairness checks.
6. Authentication and secure candidate-data handling.

## Tech stack

Python · Streamlit · scikit-learn · pandas · PyMuPDF · python-docx · GitHub
