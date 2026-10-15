HireLens AI scoring fix

This version preserves the 3-second branded intro splash and the Hire Recommendation column. It fixes overly low overall scores caused by using raw TF-IDF cosine similarity directly as a score component, while keeping raw text relevance visible separately. It also recognizes REST APIs, debugging/debug, troubleshooting, and teamwork/collaboration aliases.

Deploy: replace your repository app.py with this app.py, keep your existing requirements.txt, commit/push to GitHub, then reboot/redeploy the Streamlit app.

Note: Scores are screening aids, not validated hiring predictions. Review original resumes and avoid automatic rejection.
