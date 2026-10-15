# HireLens AI — upgraded hackathon MVP

A responsive Streamlit prototype for AI-assisted resume screening.

## What's new in this version
- Clearer workspace and navigation
- Fictional sample-resume mode
- Separate must-have and preferred skills
- Adjustable text-relevance weighting
- Search and minimum-score filters
- Sorting by match score, text relevance, or must-have coverage
- Candidate detail view with extracted source text
- Filtered CSV export
- Clear-results control
- Responsible-AI guidance

## Run locally
Use Python 3.10+.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Then:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy/update on Streamlit Community Cloud
1. Replace the files in your existing GitHub repository with these updated files.
2. Commit and push to the same branch used by Streamlit Cloud.
3. Streamlit Cloud normally redeploys automatically. If it doesn't, open the app's Manage app menu and restart/reboot it.

## Important limitations
This is a hackathon prototype, not a validated hiring system. The skill dictionary is limited, TF-IDF is not full semantic understanding, scanned PDFs may need OCR, and experience/education are not reliably validated. The score measures document relevance, not candidate quality or job-success probability. Never auto-reject applicants based on this score.

Use fictional profiles for public demos. For real candidate data, add authentication, access controls, secure storage, retention/deletion rules, fairness evaluation, and legal review.
