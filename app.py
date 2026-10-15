import re
from io import BytesIO
from datetime import datetime

import fitz
import pandas as pd
import streamlit as st
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="HireLens AI | Resume Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Expand this dictionary for the roles your demo targets.
SKILL_ALIASES = {
    "python": ["python"],
    "java": ["java"],
    "c++": ["c++"],
    "c#": ["c#"],
    "sql": ["sql", "mysql", "postgresql", "sqlite"],
    "machine learning": ["machine learning", "scikit-learn", "sklearn"],
    "deep learning": ["deep learning", "pytorch", "tensorflow", "keras"],
    "data analysis": ["data analysis", "data analytics"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "excel": ["excel", "microsoft excel"],
    "power bi": ["power bi"],
    "tableau": ["tableau"],
    "html": ["html"],
    "css": ["css"],
    "javascript": ["javascript"],
    "typescript": ["typescript"],
    "react": ["react", "react.js"],
    "node.js": ["node.js", "nodejs"],
    "django": ["django"],
    "flask": ["flask"],
    "streamlit": ["streamlit"],
    "git": ["git", "github", "gitlab"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes"],
    "rest api": ["rest api", "restful api"],
    "communication": ["communication", "verbal communication"],
    "project management": ["project management"],
    "leadership": ["leadership", "team leadership"],
    "problem solving": ["problem solving", "analytical thinking"],
    "linux": ["linux"],
    "networking": ["networking", "computer networks"],
    "cybersecurity": ["cybersecurity", "information security"],
    "mongodb": ["mongodb"],
    "firebase": ["firebase"],
    "fastapi": ["fastapi"],
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_text(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        with fitz.open(stream=data, filetype="pdf") as pdf:
            return "\n".join(page.get_text() for page in pdf)

    if filename.endswith(".docx"):
        doc = Document(BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" ".join(cell.text for cell in row.cells))
        return "\n".join(parts)

    raise ValueError("Unsupported file type. Upload PDF or DOCX.")


def find_skills(text: str) -> set:
    text = normalize(text)
    found = set()
    for skill, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            pattern = r"(?<!\w)" + re.escape(alias.lower()) + r"(?!\w)"
            if re.search(pattern, text):
                found.add(skill)
                break
    return found


def calculate_matches(job_description: str, resumes: dict, required_skills: list,
                      similarity_weight: float = 0.60) -> pd.DataFrame:
    names = list(resumes.keys())
    texts = [resumes[name] for name in names]

    # Include job description and all resumes so they share one vocabulary.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=20000,
    )
    matrix = vectorizer.fit_transform([job_description] + texts)
    similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    requested = set(required_skills) if required_skills else find_skills(job_description)
    skill_weight = 1.0 - similarity_weight
    rows = []

    for i, name in enumerate(names):
        candidate_skills = find_skills(texts[i])
        matched = requested & candidate_skills
        missing = requested - candidate_skills
        coverage = len(matched) / len(requested) if requested else None

        if requested:
            score = 100 * (
                similarity_weight * float(similarities[i])
                + skill_weight * coverage
            )
        else:
            score = 100 * float(similarities[i])

        rows.append({
            "Candidate file": name,
            "Match score (%)": round(max(0.0, min(100.0, score)), 1),
            "Text relevance (%)": round(float(similarities[i]) * 100, 1),
            "Skill coverage (%)": round(coverage * 100, 1) if coverage is not None else None,
            "Matching skills": ", ".join(sorted(matched)) or "None detected",
            "Skills to review": ", ".join(sorted(missing)) or "None detected",
            "_matched_count": len(matched),
            "_required_count": len(requested),
        })

    return pd.DataFrame(rows).sort_values(
        ["Match score (%)", "Text relevance (%)"], ascending=False
    ).reset_index(drop=True)


def score_label(score: float) -> str:
    if score >= 70:
        return "Strong document match"
    if score >= 45:
        return "Potential match — review"
    return "Lower document match — review carefully"


def load_demo_resumes():
    return {
        "Aarav_Shah_Demo.txt": (
            "Software engineering graduate. Projects in Python, SQL, pandas, "
            "machine learning and Flask. Built a student performance prediction "
            "project and a REST API. Familiar with Git and data analysis. "
            "Strong problem solving and communication."
        ),
        "Meera_Rao_Demo.txt": (
            "Computer science student with Python, scikit-learn, pandas, NumPy, "
            "deep learning and TensorFlow experience. Developed a classification "
            "model and deployed a Streamlit dashboard. Used GitHub and SQL."
        ),
        "Kabir_Nair_Demo.txt": (
            "Frontend developer with HTML, CSS, JavaScript, React and TypeScript. "
            "Built responsive web applications and collaborated with designers. "
            "Uses Git, REST APIs and problem solving."
        ),
        "Diya_Menon_Demo.txt": (
            "Data analyst intern. Experience with Excel, SQL, Power BI, Tableau, "
            "Python and pandas. Created dashboards, cleaned datasets and presented "
            "business insights to stakeholders."
        ),
    }


def inject_css():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }
      h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.035em; }
      .hero {
        padding: 1.5rem 1.6rem; border-radius: 22px;
        background: linear-gradient(120deg, #101c3a 0%, #243c79 55%, #6556c8 100%);
        color: white; margin-bottom: 1.2rem;
      }
      .hero h1 { color: white; font-size: 2.4rem; margin: 0 0 .35rem 0; }
      .hero p { color: #e2e8ff; margin: 0; font-size: 1.03rem; }
      .eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: .72rem; font-weight: 700; color: #b8c6ff; margin-bottom: .55rem; }
      .mini-card {
        border: 1px solid rgba(128, 145, 180, .28); border-radius: 16px;
        padding: 1rem 1.05rem; background: rgba(128,145,180,.05); min-height: 112px;
      }
      .mini-label { color: #8a94a8; font-size: .8rem; margin-bottom: .35rem; }
      .mini-value { font-family: 'Space Grotesk',sans-serif; font-size: 1.8rem; font-weight: 700; }
      .section-kicker { color: #7180a1; text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; font-weight: 700; }
      div[data-testid="stFileUploader"] { border-radius: 14px; }
      div.stButton > button[kind="primary"] { border-radius: 10px; font-weight: 700; min-height: 2.8rem; }
      div[data-testid="stMetric"] { border: 1px solid rgba(128,145,180,.22); padding: .8rem 1rem; border-radius: 14px; }
      .footnote { font-size: .82rem; color: #8993a8; }
    </style>
    """, unsafe_allow_html=True)


inject_css()

with st.sidebar:
    st.markdown("## 🎯 HireLens AI")
    st.caption("Resume intelligence for human-led hiring")
    st.divider()
    page = st.radio(
        "Workspace",
        ["Screen resumes", "How it works", "Responsible AI"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Demo mode**")
    use_demo = st.toggle("Load fictional sample resumes", value=False)
    st.caption("Use fictional/demo data for presentations. Do not upload sensitive candidate data to a public demo.")
    st.divider()
    st.markdown('<p class="footnote">Prototype · Python · Streamlit · scikit-learn</p>', unsafe_allow_html=True)

if page == "How it works":
    st.markdown('<div class="hero"><div class="eyebrow">Product overview</div><h1>From resume pile to reviewable shortlist.</h1><p>HireLens compares documents against a role, highlights job-related evidence, and keeps the recruiter in control.</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 01 · Upload")
        st.write("Add multiple PDF or DOCX resumes, or load fictional demo profiles.")
    with c2:
        st.markdown("### 02 · Match")
        st.write("Compare text relevance and job-related skills using a transparent baseline model.")
    with c3:
        st.markdown("### 03 · Review")
        st.write("Inspect matched skills, skills to verify, and evidence text before making a decision.")
    st.info("This MVP uses TF-IDF and a predefined skill dictionary. It is a prototype, not a validated hiring assessment.")
elif page == "Responsible AI":
    st.markdown('<div class="hero"><div class="eyebrow">Trust and transparency</div><h1>AI should support hiring decisions, not make them alone.</h1><p>Scores describe document relevance to a specific job description—not a person’s worth, ability, or future performance.</p></div>', unsafe_allow_html=True)
    st.markdown("### Built-in guardrails")
    st.markdown("""
    - **Human review:** Do not auto-reject applicants based on a score.
    - **Job-related criteria:** Use skills and role requirements that are relevant to the position.
    - **No protected-trait scoring:** Do not score candidates on age, gender, ethnicity, religion, disability, photos, or similar attributes.
    - **Data minimization:** Use only the data needed for the task and remove it when no longer required.
    - **Evidence over assumptions:** Verify qualifications and experience against the original resume.
    - **Testing:** Evaluate the system for missed candidates, parsing errors, and unfair patterns before any real-world use.
    """)
    st.warning("For a real deployment, review applicable privacy and employment laws, access controls, retention rules, and security requirements.")
else:
    st.markdown(
        '<div class="hero"><div class="eyebrow">AI-assisted talent discovery</div>'
        '<h1>Find the fit. Explain the match.</h1>'
        '<p>Screen resumes against a job description and produce a shortlist your team can actually review.</p></div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown("#### 1 · Define the role")
        role_title = st.text_input("Job title", placeholder="e.g. Junior Python Developer")
        job_description = st.text_area(
            "Job description",
            height=190,
            placeholder="Paste responsibilities, required qualifications, and relevant experience...",
        )
        default_skills = ""
        if job_description:
            detected = sorted(find_skills(job_description))
            default_skills = ", ".join(detected)
        required_skills_text = st.text_input(
            "Required skills (comma-separated)",
            value=default_skills,
            help="You can edit this list. These skills are used in the skill-coverage component.",
        )
        required_skills = list(dict.fromkeys(
            s.strip().lower() for s in required_skills_text.split(",") if s.strip()
        ))
        weight_pct = st.slider(
            "Text relevance weight",
            min_value=20,
            max_value=80,
            value=60,
            step=5,
            help="The remaining weight is assigned to required-skill coverage.",
        )

    with right:
        st.markdown("#### 2 · Add candidate resumes")
        uploaded_files = st.file_uploader(
            "Upload PDF or DOCX resumes",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="For scanned PDFs, this MVP may not extract text. OCR can be added later.",
        )
        if use_demo:
            st.success("Fictional sample profiles are enabled.")
            st.caption("Four synthetic profiles will be used for a quick demo.")
        else:
            st.caption("Supported: text-based PDF and DOCX. Files are processed in this app session; this prototype does not provide production-grade data controls.")
        st.markdown("#### What the results include")
        st.markdown("""
        - Ranked document-relevance score
        - Required-skill coverage
        - Skills detected in the resume
        - Skills to review or verify
        - Extracted text for human inspection
        """)

    run = st.button("✨ Analyze candidate matches", type="primary", use_container_width=True)

    if run:
        if not job_description.strip():
            st.error("Add a job description before running the analysis.")
        else:
            resumes = {}
            errors = []
            files_to_process = []

            if use_demo:
                resumes.update(load_demo_resumes())
            if uploaded_files:
                files_to_process.extend(uploaded_files)

            for file in files_to_process:
                try:
                    extracted = extract_text(file)
                    if not extracted.strip():
                        errors.append(f"{file.name}: no readable text found (scanned PDFs may need OCR).")
                        continue
                    name = file.name
                    suffix = 2
                    while name in resumes:
                        name = f"{suffix}_{file.name}"
                        suffix += 1
                    resumes[name] = extracted
                except Exception as exc:
                    errors.append(f"{file.name}: could not be read ({type(exc).__name__}).")

            if not resumes:
                st.error("Upload at least one readable resume or enable fictional demo resumes.")
            else:
                try:
                    with st.spinner("Comparing resumes with the role requirements..."):
                        results = calculate_matches(
                            job_description,
                            resumes,
                            required_skills,
                            similarity_weight=weight_pct / 100,
                        )
                    st.session_state["results"] = results
                    st.session_state["resume_texts"] = resumes
                    st.session_state["last_role"] = role_title or "Untitled role"
                    st.session_state["last_run"] = datetime.now().strftime("%d %b %Y, %H:%M")
                except ValueError:
                    st.error("The text could not be vectorized. Add a longer job description or more readable resume text.")

            for error in errors:
                st.warning(error)

    if "results" in st.session_state:
        results = st.session_state["results"]
        st.divider()
        st.markdown('<div class="section-kicker">Screening results</div>', unsafe_allow_html=True)
        st.header(f"Shortlist · {st.session_state.get('last_role', 'Role')}")
        st.caption(f"Last analyzed: {st.session_state.get('last_run', 'Current session')} · Review all candidates before making decisions.")

        top_score = float(results.iloc[0]["Match score (%)"]) if not results.empty else 0
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Resumes analyzed", len(results))
        m2.metric("Highest match", f"{top_score:.1f}%")
        m3.metric("Skills in criteria", int(results["_required_count"].max()) if len(results) else 0)
        m4.metric("Needs human review", len(results))

        display_cols = [
            "Candidate file", "Match score (%)", "Text relevance (%)",
            "Skill coverage (%)", "Matching skills", "Skills to review",
        ]
        st.dataframe(
            results[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Match score (%)": st.column_config.ProgressColumn(
                    "Match score (%)", min_value=0, max_value=100, format="%.1f%%"
                ),
                "Text relevance (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Skill coverage (%)": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

        candidate = st.selectbox("Inspect a candidate", results["Candidate file"].tolist())
        row = results[results["Candidate file"] == candidate].iloc[0]
        a, b = st.columns([1, 2], gap="large")
        with a:
            st.markdown("### Candidate overview")
            st.metric("Document match", f"{row['Match score (%)']}%")
            st.caption(score_label(float(row["Match score (%)"])))
            st.write("**Matching skills**")
            st.write(row["Matching skills"])
            st.write("**Skills to review**")
            st.write(row["Skills to review"])
        with b:
            st.markdown("### Resume evidence")
            resume_text = st.session_state["resume_texts"].get(candidate, "")
            st.text_area(
                "Extracted text (verify against original file)",
                value=resume_text,
                height=260,
                disabled=True,
                label_visibility="collapsed",
            )

        csv_data = results[display_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Download shortlist CSV",
            data=csv_data,
            file_name="hirelens_candidate_shortlist.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.info("The score measures text and skill overlap with this job description. It is not a probability of job success and must not be used as an automatic rejection rule.")
