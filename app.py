
import re
from datetime import datetime
from html import escape

import pymupdf
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NexaMatch | AI Resume Matcher",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. DARK DASHBOARD STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #0b1020;
        --panel: #121a2d;
        --panel-light: #18233b;
        --border: #293650;
        --text: #edf2ff;
        --muted: #94a3bd;
        --blue: #7c9cff;
        --purple: #b39aff;
        --green: #58d6a0;
        --red: #ff8295;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                ellipse at 10% 0%,
                rgba(65, 89, 180, 0.17),
                transparent 38%
            ),
            radial-gradient(
                ellipse at 95% 5%,
                rgba(128, 83, 210, 0.12),
                transparent 32%
            ),
            var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1320px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -0.4px;
    }

    p, label, li {
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, 0.85);
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 13px;
        margin-bottom: 26px;
    }

    .brand-icon {
        width: 47px;
        height: 47px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 15px;
        font-size: 25px;
        font-weight: 700;
        color: white;
        background: linear-gradient(135deg, #527bff, #a47aff);
        box-shadow: 0 8px 30px rgba(104, 117, 255, 0.22);
    }

    .brand-name {
        color: #f2f5ff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 22px;
        font-weight: 700;
        line-height: 1.25;
    }

    .brand-subtitle {
        color: var(--muted);
        font-size: 12px;
        margin-top: 4px;
    }

    .top-badge {
        display: inline-block;
        border: 1px solid #35456a;
        background: rgba(124, 156, 255, 0.08);
        color: #b9c9ff;
        border-radius: 30px;
        padding: 6px 11px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.7px;
        margin-bottom: 15px;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(30px, 4vw, 43px);
        line-height: 1.15;
        letter-spacing: -1.5px;
        font-weight: 700;
        color: #f5f7ff;
        margin: 0 0 12px 0;
    }

    .hero-gradient {
        background: linear-gradient(90deg, #83a2ff, #c0a5ff);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-description {
        max-width: 760px;
        color: #a6b3cb;
        font-size: 15px;
        line-height: 1.8;
        margin-bottom: 24px;
    }

    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 19px;
        font-weight: 600;
        color: #f1f4ff;
        margin: 8px 0 4px 0;
    }

    .section-description {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 14px;
    }

    .panel {
        background: linear-gradient(
            145deg,
            rgba(24, 35, 59, 0.94),
            rgba(16, 25, 43, 0.98)
        );
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 15px;
    }

    .panel-heading {
        color: #f0f4ff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 17px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .panel-subtitle {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.6;
        margin-bottom: 13px;
    }

    .metric-card {
        background: linear-gradient(
            145deg,
            #17233c,
            #111a2c
        );
        border: 1px solid #2a3a59;
        border-radius: 17px;
        padding: 21px;
        min-height: 145px;
    }

    .metric-label {
        color: #aab8d1;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .metric-value {
        color: #f4f6ff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 32px;
        font-weight: 700;
        letter-spacing: -1px;
        margin: 12px 0 8px 0;
    }

    .metric-note {
        color: #93a2bd;
        font-size: 11px;
        line-height: 1.5;
    }

    .skill-pill {
        display: inline-block;
        background: rgba(124, 156, 255, 0.10);
        border: 1px solid rgba(124, 156, 255, 0.27);
        color: #c7d4ff;
        border-radius: 9px;
        padding: 7px 10px;
        margin: 4px 5px 4px 0;
        font-size: 12px;
        line-height: 1.3;
    }

    .missing-pill {
        display: inline-block;
        background: rgba(255, 130, 149, 0.08);
        border: 1px solid rgba(255, 130, 149, 0.26);
        color: #ffb1bd;
        border-radius: 9px;
        padding: 7px 10px;
        margin: 4px 5px 4px 0;
        font-size: 12px;
        line-height: 1.3;
    }

    .status-box {
        border: 1px solid #2b3c5b;
        border-radius: 12px;
        padding: 13px 15px;
        background: rgba(124, 156, 255, 0.06);
        color: #b9c8e5;
        font-size: 12px;
        line-height: 1.7;
    }

    .footer {
        text-align: center;
        color: #71809c;
        font-size: 11px;
        padding-top: 22px;
        letter-spacing: 0.3px;
    }

    .stTextArea textarea {
        background: #0e1729 !important;
        color: #e9efff !important;
        border: 1px solid #34435f !important;
        border-radius: 11px !important;
        line-height: 1.6 !important;
    }

    .stTextArea textarea:focus {
        border-color: #7c9cff !important;
        box-shadow: 0 0 0 1px #7c9cff !important;
    }

    [data-testid="stFileUploader"] {
        background: #0e1729;
        border: 1px dashed #405173;
        border-radius: 12px;
        padding: 10px;
    }

    [data-testid="stFileUploader"] section {
        background: transparent;
    }

    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small {
        color: #aab8d1 !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 11px;
        border: 1px solid #6485ff;
        background: linear-gradient(100deg, #527bff, #8268f5);
        color: white;
        font-weight: 600;
        min-height: 45px;
        transition: 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #a8b8ff;
        background: linear-gradient(100deg, #648aff, #967bff);
        color: white;
        box-shadow: 0 6px 24px rgba(96, 112, 255, 0.2);
    }

    [data-testid="stMetric"] {
        background: #131e32;
        border: 1px solid #2a3a59;
        border-radius: 13px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #aab8d1 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f3f5ff !important;
    }

    [data-testid="stExpander"] {
        background: #111a2c;
        border: 1px solid #293650;
        border-radius: 12px;
    }

    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #638aff, #a783ff);
    }

    hr {
        border-color: #293650 !important;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: #91a0bb !important;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .panel {
            padding: 16px;
        }

        .metric-card {
            margin-bottom: 10px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. SKILLS DATABASE
# ============================================================

SKILL_ALIASES = {
    "python": ["python"],
    "java": ["java"],
    "c++": ["c++", "cpp"],
    "c#": ["c#"],
    "sql": ["sql"],
    "html": ["html"],
    "css": ["css"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript"],
    "react": ["react", "react.js"],
    "node.js": ["node.js", "nodejs"],
    "machine learning": ["machine learning"],
    "deep learning": ["deep learning"],
    "data analysis": [
        "data analysis",
        "data analytics",
        "data analyst",
    ],
    "data science": ["data science"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": [
        "scikit-learn",
        "sklearn",
        "scikit learn",
    ],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "excel": ["excel", "microsoft excel"],
    "git": ["git"],
    "github": ["github", "git hub"],
    "flask": ["flask"],
    "django": ["django"],
    "streamlit": ["streamlit"],
    "nlp": ["nlp", "natural language processing"],
    "statistics": ["statistics", "statistical analysis"],
    "problem solving": [
        "problem solving",
        "problem-solving",
    ],
    "communication": [
        "communication",
        "communication skills",
    ],
    "data visualization": [
        "data visualization",
        "data visualisation",
    ],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "docker": ["docker"],
    "linux": ["linux"],
    "mongodb": ["mongodb", "mongo db"],
    "mysql": ["mysql", "my sql"],
    "postgresql": ["postgresql", "postgres"],
    "rest api": [
        "rest api",
        "restful api",
        "rest apis",
    ],
    "fastapi": ["fastapi"],
    "oop": [
        "oop",
        "object-oriented programming",
        "object oriented programming",
    ],
    "data structures": [
        "data structures",
        "data structure",
    ],
    "algorithms": ["algorithms", "algorithm"],
    "linear regression": ["linear regression"],
    "logistic regression": ["logistic regression"],
    "random forest": ["random forest"],
    "gradient boosting": ["gradient boosting"],
    "decision tree": ["decision tree", "decision trees"],
    "support vector machine": [
        "support vector machine",
        "svm",
    ],
    "k-means": ["k-means", "k means", "kmeans"],
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "api development": ["api development"],
    "cybersecurity": ["cybersecurity", "cyber security"],
    "cloud computing": ["cloud computing"],
    "agile": ["agile"],
    "project management": ["project management"],
    "exploratory data analysis": [
        "exploratory data analysis",
        "eda",
    ],
    "data cleaning": ["data cleaning", "cleaning data"],
    "feature engineering": ["feature engineering"],
    "feature importance": ["feature importance"],
    "model evaluation": [
        "model evaluation",
        "model evaluation metrics",
    ],
    "jupyter notebook": [
        "jupyter notebook",
        "jupyter notebooks",
    ],
    "machine learning algorithms": [
        "machine learning algorithms",
    ],
    "teamwork": ["teamwork", "team work"],
    "leadership": ["leadership"],
    "time management": ["time management"],
    "adaptability": ["adaptability"],
}

SPECIFIC_ALGORITHMS = {
    "linear regression",
    "logistic regression",
    "random forest",
    "gradient boosting",
    "decision tree",
    "support vector machine",
    "k-means",
}


# ============================================================
# 4. LOAD SEMANTIC AI MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_semantic_model():
    """Load the sentence-transformer model once and cache it."""
    return SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# 5. PDF TEXT EXTRACTION
# ============================================================

def extract_resume_text(uploaded_file):
    """Extract readable text from an uploaded PDF."""

    pdf_bytes = uploaded_file.getvalue()

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    ) as document:
        pages = [
            page.get_text("text")
            for page in document
        ]

    return "\n".join(pages).strip()


# ============================================================
# 6. TEXT PREPROCESSING AND SKILL MATCHING
# ============================================================

def preprocess_text(text):
    """Normalize text case, punctuation variants, and whitespace."""

    text = text.lower()
    text = text.replace("\u2011", "-").replace("\u2013", "-")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def contains_phrase(text, phrase):
    """Find a phrase while avoiding accidental partial word matches."""

    escaped_phrase = re.escape(phrase.lower())
    escaped_phrase = escaped_phrase.replace(r"\ ", r"\s+")

    pattern = r"(?<!\w)" + escaped_phrase + r"(?!\w)"

    return re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    ) is not None


def extract_skills(text):
    """Extract canonical skills using aliases and explicit rules."""

    normalized_text = preprocess_text(text)
    found_skills = set()

    for canonical_skill, aliases in SKILL_ALIASES.items():
        if any(
            contains_phrase(normalized_text, alias)
            for alias in aliases
        ):
            found_skills.add(canonical_skill)

    # Recognize general algorithm knowledge if specific algorithms
    # are explicitly mentioned.
    if found_skills.intersection(SPECIFIC_ALGORITHMS):
        found_skills.add("algorithms")

    return sorted(found_skills)


# ============================================================
# 7. SIMILARITY CALCULATIONS
# ============================================================

def calculate_tfidf_similarity(resume_text, job_text):
    """Calculate lexical similarity using TF-IDF and cosine similarity."""

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    vectors = vectorizer.fit_transform(
        [resume_text, job_text]
    )

    score = cosine_similarity(
        vectors[0:1],
        vectors[1:2],
    )[0][0]

    return round(float(score) * 100, 2)


def calculate_semantic_similarity(model, resume_text, job_text):
    """Calculate semantic embedding similarity."""

    embeddings = model.encode(
        [resume_text, job_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    score = cosine_similarity(
        embeddings[0:1],
        embeddings[1:2],
    )[0][0]

    return round(float(score) * 100, 2)


# ============================================================
# 8. REPORT GENERATION
# ============================================================

def build_report(
    resume_name,
    semantic_score,
    tfidf_score,
    skill_coverage,
    matched_skills,
    missing_skills,
    resume_skills,
    job_skills,
):
    """Generate the downloadable text report."""

    matched_text = (
        "\n".join(
            f"- {skill.title()}"
            for skill in matched_skills
        )
        or "- No matching skills detected."
    )

    missing_text = (
        "\n".join(
            f"- {skill.title()}"
            for skill in missing_skills
        )
        or "- No missing recognized skills."
    )

    resume_skill_text = (
        ", ".join(
            skill.title()
            for skill in resume_skills
        )
        or "No recognized skills"
    )

    job_skill_text = (
        ", ".join(
            skill.title()
            for skill in job_skills
        )
        or "No recognized skills"
    )

    return f"""
AI-BASED RESUME AND JOB DESCRIPTION MATCHING PLATFORM
=====================================================

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Resume: {resume_name}

SIMILARITY AND COVERAGE
-----------------------
Semantic Similarity: {semantic_score:.2f}%
TF-IDF Similarity: {tfidf_score:.2f}%
Required Skill Coverage: {skill_coverage:.1f}%

MATCHING SKILLS
---------------
{matched_text}

SKILLS NOT FOUND IN RESUME
--------------------------
{missing_text}

RECOGNIZED SKILLS IN RESUME
---------------------------
{resume_skill_text}

RECOGNIZED SKILLS IN JOB DESCRIPTION
------------------------------------
{job_skill_text}

IMPORTANT NOTES
---------------
Semantic similarity measures embedding similarity between the texts.
TF-IDF measures lexical overlap in word and phrase importance.
Skill coverage measures the percentage of recognized job-description
skills that were also detected in the resume.

These metrics are not probabilities of getting hired.

The skill analyzer uses a predefined vocabulary and aliases. It can
miss context, synonyms, and qualifications expressed differently.
Algorithm-related rules are heuristic and should be reviewed manually.

Review the results before making decisions.
"""


# ============================================================
# 9. REUSABLE UI COMPONENTS
# ============================================================

def render_skill_pills(skills, missing=False):
    """Render skills as styled chips."""

    if not skills:
        st.caption("No skills to display.")
        return

    pill_class = "missing-pill" if missing else "skill-pill"

    pills = "".join(
        f'<span class="{pill_class}">{escape(skill.title())}</span>'
        for skill in skills
    )

    st.markdown(
        f'<div>{pills}</div>',
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, note):
    """Render a styled dashboard metric card."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
            <div class="metric-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 10. DASHBOARD HEADER
# ============================================================

st.markdown(
    """
    <div class="brand-row">
        <div class="brand-icon">◈</div>
        <div>
            <div class="brand-name">NexaMatch</div>
            <div class="brand-subtitle">
                AI-POWERED RESUME INTELLIGENCE
            </div>
        </div>
    </div>

    <div class="top-badge">✦ INTELLIGENT TALENT MATCHING</div>

    <div class="hero-title">
        Find the right fit with<br>
        <span class="hero-gradient">AI-powered resume analysis.</span>
    </div>

    <div class="hero-description">
        Compare a candidate's resume with a job description using
        natural language processing, semantic similarity, and
        technical skill analysis.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="status-box">
        <b>Privacy note:</b> This application processes the uploaded
        resume in memory. This code does not save resumes to a database
        or write them to a file.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.divider()


# ============================================================
# 11. INPUT WORKSPACE
# ============================================================

st.markdown(
    '<div class="section-label">Analysis workspace</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-description">
        Upload a resume and provide the target job description
        to begin your analysis.
    </div>
    """,
    unsafe_allow_html=True,
)

resume_col, job_col = st.columns([0.9, 1.1], gap="large")

with resume_col:
    st.markdown(
        """
        <div class="panel-heading">📄 Candidate resume</div>
        <div class="panel-subtitle">
            Upload a PDF resume for text extraction and skill analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_resume = st.file_uploader(
        "Upload PDF resume",
        type=["pdf"],
        help="Text-based PDFs work best. Scanned PDFs may need OCR.",
        key="resume_uploader",
    )

    if uploaded_resume is not None:
        st.success(
            f"Resume selected: {uploaded_resume.name}"
        )

with job_col:
    st.markdown(
        """
        <div class="panel-heading">💼 Target job description</div>
        <div class="panel-subtitle">
            Paste the job requirements, responsibilities, and desired skills.
        </div>
        """,
        unsafe_allow_html=True,
    )

    job_description = st.text_area(
        "Job description",
        height=235,
        placeholder=(
            "Example: We are looking for a Data Analyst Intern "
            "with Python, SQL, Pandas, NumPy, data visualization, "
            "and analytical problem-solving skills..."
        ),
        label_visibility="collapsed",
        key="job_description",
    )

st.write("")

analyze_col, reset_col = st.columns([3, 1])

with analyze_col:
    analyze_clicked = st.button(
        "✦  Analyze Resume",
        type="primary",
        use_container_width=True,
    )

with reset_col:
    reset_clicked = st.button(
        "Clear analysis",
        use_container_width=True,
    )


if reset_clicked:
    st.session_state.pop("analysis_result", None)
    st.rerun()


# ============================================================
# 12. RUN ANALYSIS
# ============================================================

if analyze_clicked:

    if uploaded_resume is None:
        st.error("Please upload a PDF resume.")

    elif not job_description.strip():
        st.error("Please enter a job description.")

    else:
        try:
            with st.spinner("Extracting resume text..."):
                resume_text = extract_resume_text(uploaded_resume)

            if not resume_text:
                st.error(
                    "No readable text was found in this PDF. "
                    "Try a text-based PDF instead of a scanned image."
                )
                st.stop()

            resume_clean = preprocess_text(resume_text)
            job_clean = preprocess_text(job_description)

            # -----------------------------------------------
            # TF-IDF similarity
            # -----------------------------------------------

            with st.spinner("Calculating text similarity..."):
                try:
                    tfidf_score = calculate_tfidf_similarity(
                        resume_clean,
                        job_clean,
                    )
                except ValueError:
                    tfidf_score = 0.0
                    st.warning(
                        "TF-IDF could not be calculated from this text. "
                        "A score of 0.00% is shown."
                    )

            # -----------------------------------------------
            # Semantic similarity
            # -----------------------------------------------

            with st.spinner(
                "Loading semantic AI and analyzing meaning. "
                "The first run may take longer..."
            ):
                model = load_semantic_model()

                semantic_score = calculate_semantic_similarity(
                    model,
                    resume_text,
                    job_description,
                )

            # -----------------------------------------------
            # Skill extraction
            # -----------------------------------------------

            resume_skills = set(
                extract_skills(resume_text)
            )

            job_skills = set(
                extract_skills(job_description)
            )

            matched_skills = sorted(
                resume_skills & job_skills
            )

            missing_skills = sorted(
                job_skills - resume_skills
            )

            skill_coverage = (
                len(matched_skills) / len(job_skills) * 100
                if job_skills
                else 0.0
            )

            # -----------------------------------------------
            # Store results so they persist across reruns
            # -----------------------------------------------

            st.session_state["analysis_result"] = {
                "resume_name": uploaded_resume.name,
                "resume_text": resume_text,
                "job_description": job_description,
                "semantic_score": semantic_score,
                "tfidf_score": tfidf_score,
                "skill_coverage": skill_coverage,
                "resume_skills": sorted(resume_skills),
                "job_skills": sorted(job_skills),
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
            }

        except Exception as error:
            st.error(
                "An error occurred during analysis. "
                "Check the details below."
            )
            st.exception(error)


# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

result = st.session_state.get("analysis_result")

if result is not None:

    st.divider()

    st.markdown(
        """
        <div class="top-badge">✓ ANALYSIS COMPLETE</div>
        <div class="hero-title" style="font-size:30px;">
            Resume analysis <span class="hero-gradient">overview</span>
        </div>
        <div class="hero-description">
            Review the similarity metrics and compare the detected skills
            against the target job description.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Analyzed resume: {result['resume_name']}"
    )

    # --------------------------------------------------------
    # Metric cards
    # --------------------------------------------------------

    metric_cols = st.columns(3, gap="medium")

    with metric_cols[0]:
        render_metric_card(
            "Semantic similarity",
            f"{result['semantic_score']:.2f}%",
            "Meaning and contextual similarity",
        )

    with metric_cols[1]:
        render_metric_card(
            "TF-IDF similarity",
            f"{result['tfidf_score']:.2f}%",
            "Overlap in weighted terms",
        )

    with metric_cols[2]:
        render_metric_card(
            "Required skill coverage",
            f"{result['skill_coverage']:.1f}%",
            "Recognized job skills also found in resume",
        )

    st.write("")

    st.markdown(
        '<div class="section-label">Similarity visualization</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "These metrics measure different aspects of similarity. "
        "They are not probabilities of being hired."
    )

    st.write("Semantic similarity")
    st.progress(
        min(max(result["semantic_score"] / 100, 0.0), 1.0)
    )

    st.write("TF-IDF similarity")
    st.progress(
        min(max(result["tfidf_score"] / 100, 0.0), 1.0)
    )

    st.write("Required skill coverage")
    st.progress(
        min(max(result["skill_coverage"] / 100, 0.0), 1.0)
    )

    # --------------------------------------------------------
    # Skills comparison
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-label">Skills comparison</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Skills are extracted using a predefined vocabulary and aliases.
            Review the results manually.
        </div>
        """,
        unsafe_allow_html=True,
    )

    matched_col, missing_col = st.columns(2, gap="large")

    with matched_col:
        st.markdown(
            f"""
            <div class="panel-heading">
                ✓ Matching skills ({len(result['matched_skills'])})
            </div>
            <div class="panel-subtitle">
                Recognized in both the resume and job description.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result["matched_skills"]:
            render_skill_pills(result["matched_skills"])
        else:
            st.info("No matching skills were detected.")

    with missing_col:
        st.markdown(
            f"""
            <div class="panel-heading">
                ! Skills not found ({len(result['missing_skills'])})
            </div>
            <div class="panel-subtitle">
                Recognized in the job description but not detected in the resume.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result["missing_skills"]:
            render_skill_pills(
                result["missing_skills"],
                missing=True,
            )
        elif result["job_skills"]:
            st.success(
                "All recognized job-description skills were detected."
            )
        else:
            st.info(
                "No recognized skills were found in the job description."
            )

    # --------------------------------------------------------
    # Skill summary
    # --------------------------------------------------------

    st.divider()

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.metric(
            "Skills detected in resume",
            len(result["resume_skills"]),
        )

    with summary_col2:
        st.metric(
            "Skills detected in job description",
            len(result["job_skills"]),
        )

    # --------------------------------------------------------
    # Downloadable report
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-label">Export your results</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Download a text report containing the metrics, matched skills,
            missing skills, and recognized skill lists.
        </div>
        """,
        unsafe_allow_html=True,
    )

    report = build_report(
        resume_name=result["resume_name"],
        semantic_score=result["semantic_score"],
        tfidf_score=result["tfidf_score"],
        skill_coverage=result["skill_coverage"],
        matched_skills=result["matched_skills"],
        missing_skills=result["missing_skills"],
        resume_skills=result["resume_skills"],
        job_skills=result["job_skills"],
    )

    st.download_button(
        label="⬇  Download analysis report",
        data=report,
        file_name="resume_analysis_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Expandable details
    # --------------------------------------------------------

    with st.expander("View extracted resume text"):
        st.text(result["resume_text"])

    with st.expander("View all recognized skills"):
        st.markdown("**Skills detected in resume**")
        render_skill_pills(result["resume_skills"])

        st.markdown("**Skills detected in job description**")
        render_skill_pills(result["job_skills"])

    st.markdown(
        """
        <div class="status-box">
            <b>How to interpret these results</b><br>
            Semantic similarity compares text embeddings. TF-IDF measures
            lexical overlap. Skill coverage measures overlap among recognized
            skills. Skill extraction is rule-based and may miss synonyms,
            context, or qualifications expressed differently.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 14. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        NEXAMATCH · AI RESUME INTELLIGENCE<br>
        Python · Streamlit · NLP · Sentence Transformers
    </div>
    """,
    unsafe_allow_html=True,
)
