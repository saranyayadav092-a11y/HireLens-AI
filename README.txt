HireLens AI — Updated scoring version

Included:
- app.py: your existing app with the branded 3-second intro splash, IST timestamp,
  recommendation column, candidate details, and CSV export preserved.
- Updated scoring: raw TF-IDF cosine similarity is square-root calibrated for the
  overall score component, while the Text relevance column continues to show the
  original raw similarity. Must-have and preferred skill coverage remain explicit.

Deploy:
1. Replace the app.py in your GitHub repository with this app.py.
2. Keep your existing requirements.txt (the app dependencies have not changed).
3. Commit and push to the same branch configured in Streamlit Community Cloud.
4. Wait for redeploy, then run a fresh screening. Old results in an existing session
   will not recalculate until you analyze the resumes again.

Important:
The square-root calibration is a transparent heuristic to reduce the domination of
very small raw TF-IDF values; it is not a trained or validated hiring model. It may
change scores and ordering. Review the formula against representative resumes and
never use the score as an automatic hire/reject decision.
