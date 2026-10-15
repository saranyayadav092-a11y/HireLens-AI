HireLens AI — Hire Recommendation + Opening Splash

This version keeps the branded 3-second opening splash screen and the Hire Recommendation column beside each candidate name.

To use:
1. Replace your existing app.py with this app.py.
2. Keep your existing requirements.txt.
3. Commit/push the change to GitHub and redeploy/reboot the Streamlit app.

The splash is shown once per Streamlit session. The recommendation is a preliminary resume-screening aid only; review the original resume and make hiring decisions with a human recruiter.


Recommendation logic update: candidates are now grouped by match score (70%+ strong match, 45–69.9% potential match, below 45% lower match). Missing must-have terms are flagged for human verification and do not automatically disqualify a candidate. These are screening aids, not hiring decisions. The 3-second opening splash remains included.
