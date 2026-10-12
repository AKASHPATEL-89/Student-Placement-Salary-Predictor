import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
import random

# ==================================================================
# PAGE CONFIG
# ==================================================================
st.set_page_config(
    page_title="🎓 Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================================
# BACKGROUND IMAGE LIBRARY (Pexels/Pixabay — free to use)
# ==================================================================
BACKGROUNDS = [
    # 1. Tech / Coding
    {
        "url": "https://images.pexels.com/photos/577585/pexels-photo-577585.jpeg?auto=compress&cs=tinysrgb&w=1920",
        "overlay": "rgba(15, 23, 42, 0.72)",
        "accent": "#38bdf8",
        "is_dark": True,
    },
    # 2. University Campus
    {
        "url": "https://images.pexels.com/photos/207692/pexels-photo-207692.jpeg?auto=compress&cs=tinysrgb&w=1920",
        "overlay": "rgba(30, 41, 59, 0.75)",
        "accent": "#818cf8",
        "is_dark": True,
    },
    # 3. Books / Study
    {
        "url": "https://images.pexels.com/photos/159711/books-bookstore-book-reading-159711.jpeg?auto=compress&cs=tinysrgb&w=1920",
        "overlay": "rgba(15, 23, 42, 0.78)",
        "accent": "#f472b6",
        "is_dark": True,
    },
    # 4. Graduation
    {
        "url": "https://images.pexels.com/photos/267885/pexels-photo-267885.jpeg?auto=compress&cs=tinysrgb&w=1920",
        "overlay": "rgba(30, 27, 75, 0.75)",
        "accent": "#a78bfa",
        "is_dark": True,
    },
    # 5. Office / Corporate
    {
        "url": "https://images.pexels.com/photos/380769/pexels-photo-380769.jpeg?auto=compress&cs=tinysrgb&w=1920",
        "overlay": "rgba(15, 23, 42, 0.75)",
        "accent": "#34d399",
        "is_dark": True,
    },
    # 6. Laptop Workspace
    {
        "url": "https://images.pexels.com/photos/205421/pexels-photo-205421.jpeg?auto=compress&cs=tinysrgb&w=1920",
        "overlay": "rgba(15, 23, 42, 0.72)",
        "accent": "#fbbf24",
        "is_dark": True,
    },
    # 7. Data / Analytics
    {
        "url": "https://images.pexels.com/photos/669610/pexels-photo-669610.jpeg?auto=compress&cs=tinysrgb&w=1920",
        "overlay": "rgba(12, 74, 110, 0.78)",
        "accent": "#22d3ee",
        "is_dark": True,
    },
    # 8. Library
    {
        "url": "https://images.pexels.com/photos/590493/pexels-photo-590493.jpeg?auto=compress&cs=tinysrgb&w=1920",
        "overlay": "rgba(45, 27, 105, 0.78)",
        "accent": "#c084fc",
        "is_dark": True,
    },
]

# Pick a random theme on first load
if "bg_idx" not in st.session_state:
    st.session_state.bg_idx = random.randint(0, len(BACKGROUNDS) - 1)

THEME       = BACKGROUNDS[st.session_state.bg_idx]
BG_URL      = THEME["url"]
OVERLAY     = THEME["overlay"]
ACCENT      = THEME["accent"]
TEXT_COLOR  = "#ffffff"
CARD_BG     = "rgba(15, 23, 42, 0.65)"
CARD_BORDER = "rgba(255, 255, 255, 0.15)"


# ==================================================================
# GLOBAL CSS + BACKGROUND IMAGE
# ==================================================================
st.markdown(f"""
<style>
    /* ====== BACKGROUND IMAGE with overlay ====== */
    .stApp {{
        background-image:
            linear-gradient({OVERLAY}, {OVERLAY}),
            url("{BG_URL}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}

    /* ====== MAIN CONTAINER ====== */
    .main .block-container {{
        position: relative;
        z-index: 1;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }}

    /* ====== TEXT ====== */
    h1, h2, h3, h4, h5, h6, p, label, span, div, .stMarkdown {{
        color: {TEXT_COLOR} !important;
    }}

    /* ====== HEADER BANNER ====== */
    .header-banner {{
        background: rgba(15, 23, 42, 0.55);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        padding: 2.5rem 2rem;
        border-radius: 24px;
        border: 1px solid {CARD_BORDER};
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 12px 48px rgba(0,0,0,0.4);
    }}
    .header-banner h1 {{
        font-size: 2.6rem !important;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #ffffff, {ACCENT});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .header-banner p {{
        margin-top: 0.7rem;
        font-size: 1.05rem;
        color: #e2e8f0 !important;
        opacity: 0.95;
    }}

    /* ====== GLASS CARDS ====== */
    .glass-card {{
        background: {CARD_BG};
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 1.5rem;
        border-radius: 18px;
        border: 1px solid {CARD_BORDER};
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-bottom: 1.2rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    .glass-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.45);
    }}
    .glass-card h3 {{
        margin: 0 0 0.6rem 0;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: {ACCENT} !important;
        font-weight: 700;
    }}
    .glass-card h2 {{
        margin: 0;
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff !important;
    }}

    /* ====== METRIC CARDS ====== */
    .metric-card {{
        background: rgba(15, 23, 42, 0.55);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        padding: 1.1rem 0.8rem;
        border-radius: 16px;
        border: 1px solid {CARD_BORDER};
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        transition: all 0.3s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-4px);
        border-color: {ACCENT};
        box-shadow: 0 10px 32px {ACCENT}55;
    }}
    .metric-label {{
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #cbd5e1 !important;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }}
    .metric-value {{
        font-size: 1.4rem;
        font-weight: 800;
        color: {ACCENT} !important;
        text-shadow: 0 0 24px {ACCENT}88;
    }}

    /* ====== SECTION HEADINGS ====== */
    .section-head {{
        font-size: 1.3rem;
        font-weight: 800;
        color: #ffffff !important;
        padding: 0.5rem 0;
        margin: 1rem 0;
        border-bottom: 3px solid {ACCENT};
        display: inline-block;
        letter-spacing: -0.3px;
    }}

    /* ====== SKILL BADGES ====== */
    .badge {{
        display: inline-block;
        padding: 7px 16px;
        border-radius: 24px;
        font-size: 0.85rem;
        font-weight: 700;
        margin: 4px 5px 4px 0;
        border: 1px solid rgba(255,255,255,0.25);
        backdrop-filter: blur(10px);
    }}
    .badge-yes {{
        background: linear-gradient(135deg, rgba(16,185,129,0.95), rgba(5,150,105,0.95));
        color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(16,185,129,0.5);
    }}
    .badge-no {{
        background: linear-gradient(135deg, rgba(239,68,68,0.95), rgba(220,38,38,0.95));
        color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(239,68,68,0.5);
    }}

    /* ====== RESULT BOXES ====== */
    .result-success {{
        background: linear-gradient(135deg, rgba(16,185,129,0.95), rgba(5,150,105,0.95));
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: #ffffff !important;
        box-shadow: 0 12px 40px rgba(16,185,129,0.5);
        animation: pulseGlow 2s ease infinite;
    }}
    .result-danger {{
        background: linear-gradient(135deg, rgba(239,68,68,0.95), rgba(220,38,38,0.95));
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: #ffffff !important;
        box-shadow: 0 12px 40px rgba(239,68,68,0.5);
    }}
    .result-success h2, .result-danger h2,
    .result-success p, .result-danger p {{
        color: #ffffff !important;
        margin: 0;
    }}
    @keyframes pulseGlow {{
        0%, 100% {{ box-shadow: 0 12px 40px rgba(16,185,129,0.5); }}
        50%      {{ box-shadow: 0 12px 55px rgba(16,185,129,0.85); }}
    }}

    /* ====== SIDEBAR ====== */
    section[data-testid="stSidebar"] {{
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-right: 1px solid {CARD_BORDER};
    }}
    section[data-testid="stSidebar"] * {{
        color: #f1f5f9 !important;
    }}
    section[data-testid="stSidebar"] .stCaption {{
        color: #94a3b8 !important;
    }}

    /* ====== BUTTONS ====== */
    .stButton > button {{
        background: linear-gradient(135deg, {ACCENT}, {ACCENT}cc);
        color: #ffffff !important;
        border: none;
        border-radius: 14px;
        padding: 0.85rem 2rem;
        font-weight: 800;
        font-size: 1rem;
        letter-spacing: 0.5px;
        box-shadow: 0 8px 24px {ACCENT}88;
        transition: all 0.3s ease;
        width: 100%;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 32px {ACCENT}cc;
        filter: brightness(1.1);
    }}

    /* ====== INPUTS ====== */
    .stSelectbox > div > div,
    .stNumberInput input {{
        background: rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.20) !important;
        color: #ffffff !important;
    }}
    .stSlider [data-baseweb="slider"] {{
        padding-top: 6px;
    }}

    /* ====== EXPANDER ====== */
    .streamlit-expanderHeader {{
        background: rgba(255,255,255,0.06) !important;
        border-radius: 12px;
        color: #ffffff !important;
        font-weight: 700;
    }}
    .streamlit-expanderContent {{
        background: rgba(15, 23, 42, 0.5) !important;
        border-radius: 0 0 12px 12px;
    }}

    /* ====== ALERTS ====== */
    .stAlert {{
        border-radius: 14px !important;
        backdrop-filter: blur(12px);
    }}

    /* ====== FOOTER ====== */
    .custom-footer {{
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(18px);
        border-radius: 16px;
        border: 1px solid {CARD_BORDER};
        font-size: 0.9rem;
    }}
    .custom-footer p {{
        margin: 0.4rem 0;
    }}
    .accent-text {{
        color: {ACCENT} !important;
        font-weight: 700;
    }}

    /* ====== DATAFRAME ====== */
    .stDataFrame {{
        background: rgba(15, 23, 42, 0.55) !important;
        border-radius: 12px !important;
    }}
</style>
""", unsafe_allow_html=True)


# ==================================================================
# LOAD MODELS
# ==================================================================
@st.cache_resource
def load_models():
    clf = joblib.load("models/best_classifier.pkl")
    reg = joblib.load("models/best_regressor.pkl")
    return clf, reg

try:
    clf, reg = load_models()
except Exception as e:
    st.error(f"⚠️ Model loading failed: {e}")
    st.stop()


# ==================================================================
# FEATURE ENGINEERING
# ==================================================================
def add_features(df):
    df = df.copy()
    df["skill_total"]   = df[["python_skill","dsa_skill","ml_skill","web_dev_skill"]].sum(axis=1)
    df["coding_dsa"]    = df["coding_score"] * df["dsa_skill"]
    df["cgpa_tier"]     = df["cgpa"] / df["college_tier"]
    df["proj_intern"]   = df["projects"] + df["internships"] * 2
    df["resume_comm"]   = df["resume_score"] * df["communication_score"]
    df["backlog_pen"]   = df["backlogs"] * -1
    df["aptitude_cgpa"] = df["aptitude_score"] * df["cgpa"] / 10
    return df


# ==================================================================
# HEADER
# ==================================================================
st.markdown("""
<div class="header-banner">
    <h1>🎓 Student Placement & Salary Predictor</h1>
    <p>⚡ AI-Powered Career Prediction · Logistic Regression + Random Forest</p>
</div>
""", unsafe_allow_html=True)


# ==================================================================
# SIDEBAR
# ==================================================================
with st.sidebar:
    st.markdown("## 🎯 Student Profile")
    st.markdown("Configure the candidate's details below.")
    st.markdown("---")

    with st.expander("🎓 Academic Details", expanded=True):
        cgpa   = st.slider("CGPA", 5.0, 10.0, 7.5, 0.01)
        branch = st.selectbox("Branch", ["CSE","IT","ECE","EEE","Mechanical","Civil"])
        tier   = st.selectbox("College Tier", [1, 2, 3],
                              format_func=lambda x: f"Tier {x}")
        backlogs = st.number_input("Backlogs", 0, 3, 0)

    with st.expander("💻 Technical Skills", expanded=True):
        st.caption("**1 = Yes (has skill)** · **0 = No (does not have skill)**")
        python_skill  = st.selectbox("Python",  [0, 1],
                                     format_func=lambda x: "✓ Yes (Proficient)" if x else "✗ No ")
        dsa_skill     = st.selectbox("DSA",     [0, 1],
                                     format_func=lambda x: "✓ Yes (Practiced)"  if x else "✗ No ")
        ml_skill      = st.selectbox("ML",      [0, 1],
                                     format_func=lambda x: "✓ Yes (Trained)"    if x else "✗ No ")
        web_dev_skill = st.selectbox("Web Dev", [0, 1],
                                     format_func=lambda x: "✓ Yes (Built apps)" if x else "✗ No ")

    with st.expander("📊 Performance Scores", expanded=True):
        coding_score  = st.slider("Coding Score",  0.0, 100.0, 60.0)
        communication = st.slider("Communication", 4.0, 10.0,  7.0)
        aptitude      = st.slider("Aptitude",     40.0, 100.0, 70.0)

    with st.expander("🚀 Experience", expanded=True):
        internships = st.number_input("Internships", 0, 3, 1)
        projects    = st.number_input("Projects",    1, 6, 3)

    with st.expander("📄 Resume", expanded=True):
        resume_score = st.slider("Resume Score", 16.5, 132.0, 70.0)
        skill_score  = st.slider("Skill Score",  0, 4, 2)

    st.markdown("---")
    if st.button("🎲 Change Background"):
        st.session_state.bg_idx = random.randint(0, len(BACKGROUNDS) - 1)
        st.rerun()

    st.caption(f"🎨 Theme {st.session_state.bg_idx + 1} of {len(BACKGROUNDS)}")


# ==================================================================
# BUILD INPUT
# ==================================================================
raw = pd.DataFrame([{
    "cgpa": cgpa, "branch": branch, "college_tier": tier,
    "python_skill": python_skill, "dsa_skill": dsa_skill,
    "ml_skill": ml_skill, "web_dev_skill": web_dev_skill,
    "coding_score": coding_score, "communication_score": communication,
    "aptitude_score": aptitude, "internships": internships,
    "projects": projects, "backlogs": backlogs,
    "resume_score": resume_score, "skill_score": skill_score,
}])
df_input = add_features(raw)


# ==================================================================
# LAYOUT
# ==================================================================
col_left, col_right = st.columns([1, 1], gap="large")

# ---------- LEFT ----------
with col_left:
    st.markdown('<div class="section-head">📄 Profile Overview</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, (lbl, val) in zip([c1, c2, c3], [("CGPA", f"{cgpa:.2f}"),
                                                ("Tier",  f"{tier}"),
                                                ("Branch", branch)]):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    for col, (lbl, val) in zip([c4, c5, c6], [("Coding", f"{coding_score:.0f}"),
                                                ("Aptitude", f"{aptitude:.0f}"),
                                                ("Backlogs", str(backlogs))]):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">💡 Skill Set</div>', unsafe_allow_html=True)
    badges = ""
    for name, val in [("Python", python_skill), ("DSA", dsa_skill),
                       ("ML", ml_skill), ("Web Dev", web_dev_skill)]:
        cls  = "badge-yes" if val else "badge-no"
        icon = "✓" if val else "✗"
        badges += f'<span class="badge {cls}">{icon} {name}</span>'
    st.markdown(f'<div style="margin-bottom:1rem;">{badges}</div>', unsafe_allow_html=True)

    with st.expander("🔬 Engineered Features (Model Input)"):
        feats = {
            "Skill Total":      int(df_input["skill_total"].iloc[0]),
            "Coding × DSA":     float(df_input["coding_dsa"].iloc[0]),
            "CGPA / Tier":      round(float(df_input["cgpa_tier"].iloc[0]), 2),
            "Projects + 2×Int": int(df_input["proj_intern"].iloc[0]),
            "Resume × Comm":    float(df_input["resume_comm"].iloc[0]),
            "Backlog Penalty":  int(df_input["backlog_pen"].iloc[0]),
            "Aptitude × CGPA":  round(float(df_input["aptitude_cgpa"].iloc[0]), 2),
        }
        st.dataframe(pd.DataFrame(feats.items(), columns=["Feature", "Value"]),
                     use_container_width=True, hide_index=True)


# ---------- RIGHT ----------
with col_right:
    st.markdown('<div class="section-head">🔮 Prediction Engine</div>', unsafe_allow_html=True)

    if st.button("🚀 Predict Placement & Salary", use_container_width=True):
        placed = clf.predict(df_input)[0]
        prob   = clf.predict_proba(df_input)[0][1]

        # Result banner
        if placed == 1:
            st.markdown("""
            <div class="result-success">
                <h2 style="font-size:2rem;">✅ Placed</h2>
                <p style="margin-top:0.5rem; opacity:0.95;">Congratulations! This candidate is likely to be placed.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-danger">
                <h2 style="font-size:2rem;">❌ Not Placed</h2>
                <p style="margin-top:0.5rem; opacity:0.95;">This candidate needs profile improvement.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 38, "color": "#ffffff"}},
            delta={"reference": 50, "increasing": {"color": "#10b981"},
                                    "decreasing": {"color": "#ef4444"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#ffffff",
                          "tickfont": {"color": "#ffffff"}},
                "bar": {"color": ACCENT, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0.3)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40],   "color": "rgba(239,68,68,0.25)"},
                    {"range": [40, 70],  "color": "rgba(251,191,36,0.25)"},
                    {"range": [70, 100], "color": "rgba(16,185,129,0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 4},
                    "thickness": 0.85, "value": 50,
                },
            },
        ))
        fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ffffff", "family": "sans-serif"},
        )
        st.plotly_chart(fig, use_container_width=True)

        # Salary or suggestions
        if placed == 1:
            salary = reg.predict(df_input)[0]
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; background: linear-gradient(135deg, rgba(102,126,234,0.85), rgba(118,75,162,0.85));">
                <h3 style="color:#fbbf24 !important;">💰 Expected Salary</h3>
                <h2 style="color:#ffffff !important; font-size:2.6rem;">₹ {salary:.2f} LPA</h2>
                <p style="color:#e2e8f0 !important; opacity:0.9; margin-top:0.5rem;">
                    Based on the training dataset's distribution
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card">
                <h3>💡 Improvement Suggestions</h3>
                <p style="margin:0.5rem 0; color:#e2e8f0 !important;">• Strengthen at least 2 technical skills</p>
                <p style="margin:0.5rem 0; color:#e2e8f0 !important;">• Push coding score above 70</p>
                <p style="margin:0.5rem 0; color:#e2e8f0 !important;">• Complete 2+ internships, build 4+ projects</p>
                <p style="margin:0.5rem 0; color:#e2e8f0 !important;">• Maintain CGPA above 7.5 with zero backlogs</p>
                <p style="margin:0.5rem 0; color:#e2e8f0 !important;">• Polish resume to 80+ score</p>
            </div>
            """, unsafe_allow_html=True)

        # Interpretation
        if prob >= 0.85:
            st.success(f"🎯 High confidence ({prob*100:.1f}%) — Strong placement likelihood")
        elif prob >= 0.65:
            st.info(f"👍 Moderate confidence ({prob*100:.1f}%) — Good chances, polish weak areas")
        elif prob >= 0.40:
            st.warning(f"⚠️ Borderline ({prob*100:.1f}%) — Meaningful improvement needed")
        else:
            st.error(f"❌ Low confidence ({prob*100:.1f}%) — Significant gaps to fill")


# ==================================================================
# FOOTER
# ==================================================================
st.markdown(f"""
<div class="custom-footer">
    <p style="font-weight:700;">🎓 <span class="accent-text">Student Placement Predictor</span></p>
    <p style="font-size:0.85rem; opacity:0.85;">
        ⚡ Built with Streamlit · Scikit-learn · XGBoost · Plotly
    </p>
    <p style="font-size:0.78rem; opacity:0.7;">
        🕐 {datetime.now().strftime('%d %b %Y · %H:%M')} &nbsp;·&nbsp; 🎨 Theme #{st.session_state.bg_idx + 1} of {len(BACKGROUNDS)}
    </p>
</div>
""", unsafe_allow_html=True)