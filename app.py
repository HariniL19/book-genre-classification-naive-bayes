"""Book Genre Classification Using Naive Bayes
College Information Retrieval (IR) Project
Built with Streamlit, Scikit-learn, Pandas, and Plotly.
Modern UI/UX Edition with Premium Glassmorphism & Interactive Math Engine.
"""

import os
import json
import random
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Book Genre Classifier | Naive Bayes IR Project",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern UI/UX Design System (CSS)
# ---------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">

<style>
    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }

    /* Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        padding: 26px 30px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15), 0 8px 10px -6px rgba(15, 23, 42, 0.1);
        position: relative;
        overflow: hidden;
    }

    .hero-card::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -20%;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.25) 0%, rgba(147, 51, 234, 0.05) 70%, transparent 100%);
        border-radius: 50%;
        pointer-events: none;
    }

    .project-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(59, 130, 246, 0.18);
        border: 1px solid rgba(96, 165, 250, 0.4);
        color: #93C5FD;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 4px 14px;
        border-radius: 30px;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        line-height: 1.2;
        margin-bottom: 8px;
        background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 60%, #93C5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-desc {
        font-size: 0.98rem;
        color: #94A3B8;
        line-height: 1.55;
        max-width: 820px;
        margin-bottom: 0px;
    }

    /* Glass Cards */
    .glass-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .glass-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
    }

    /* Prediction Result Showcase */
    .winner-card {
        border-radius: 16px;
        padding: 24px;
        color: white;
        position: relative;
        overflow: hidden;
        margin-bottom: 18px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.15);
    }

    .winner-genre-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 4px 0 8px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .confidence-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Keyword Tags */
    .token-tag {
        display: inline-block;
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 3px 4px 3px 0;
    }

    .token-count {
        background: #DBEAFE;
        color: #1E40AF;
        padding: 1px 6px;
        border-radius: 8px;
        font-size: 0.75rem;
        margin-left: 4px;
    }

    /* Metric Box */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    .kpi-label {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #E2E8F0;
        margin-bottom: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #64748B;
        background-color: transparent;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        color: #1D4ED8 !important;
        background-color: #EFF6FF !important;
        border-bottom: 3px solid #2563EB !important;
    }

    /* Custom Button styling */
    div.stButton > button:first-child {
        border-radius: 10px;
        font-weight: 700;
        letter-spacing: -0.01em;
        transition: all 0.2s ease;
    }

    /* Quick chip buttons */
    .quick-chip {
        display: inline-block;
        padding: 6px 14px;
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #334155;
        margin-right: 8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Metadata & Palettes for the 8 Target Genres
# ---------------------------------------------------------
GENRE_THEMES = {
    "Fantasy": {
        "icon": "🧙‍♂️",
        "color": "#7C3AED",
        "gradient": "linear-gradient(135deg, #6D28D9 0%, #8B5CF6 100%)",
        "bg_light": "#F5F3FF",
        "desc": "High magic systems, mythical beasts, ancient prophecies, arcane artifacts, and fantastical kingdoms."
    },
    "Science Fiction": {
        "icon": "🛸",
        "color": "#0284C7",
        "gradient": "linear-gradient(135deg, #0369A1 0%, #06B6D4 100%)",
        "bg_light": "#F0F9FF",
        "desc": "Cybernetics, artificial intelligence, interstellar spaceflight, wormholes, and futuristic societies."
    },
    "Mystery": {
        "icon": "🔍",
        "color": "#2563EB",
        "gradient": "linear-gradient(135deg, #1D4ED8 0%, #3B82F6 100%)",
        "bg_light": "#EFF6FF",
        "desc": "Detectives, forensic investigations, locked-room puzzles, poisonings, and deductive logic."
    },
    "Romance": {
        "icon": "💖",
        "color": "#DB2777",
        "gradient": "linear-gradient(135deg, #BE185D 0%, #EC4899 100%)",
        "bg_light": "#FDF2F8",
        "desc": "Intense romantic chemistry, emotional vulnerability, second chances, and passionate devotion."
    },
    "Horror": {
        "icon": "🩸",
        "color": "#DC2626",
        "gradient": "linear-gradient(135deg, #B91C1C 0%, #EF4444 100%)",
        "bg_light": "#FEF2F2",
        "desc": "Supernatural dread, haunted estates, demonic entities, psychological torment, and gruesome terror."
    },
    "Thriller": {
        "icon": "⚡",
        "color": "#D97706",
        "gradient": "linear-gradient(135deg, #B45309 0%, #F59E0B 100%)",
        "bg_light": "#FFFBEB",
        "desc": "High-stakes espionage, ticking-clock crises, assassins, undercover assets, and lethal conspiracies."
    },
    "Historical Fiction": {
        "icon": "📜",
        "color": "#B45309",
        "gradient": "linear-gradient(135deg, #78350F 0%, #B45309 100%)",
        "bg_light": "#FFFBEB",
        "desc": "Authentic period detail, world wars, royal dynasties, revolutions, and historical chronicles."
    },
    "Fiction": {
        "icon": "📖",
        "color": "#059669",
        "gradient": "linear-gradient(135deg, #047857 0%, #10B981 100%)",
        "bg_light": "#ECFDF5",
        "desc": "Contemporary, literary character studies, domestic drama, family identity, and societal reflections."
    }
}

# ---------------------------------------------------------
# Curated Sample Books for Instant Testing (All 8 Genres)
# ---------------------------------------------------------
PRESET_SAMPLES = [
    {
        "genre": "Mystery",
        "title": "The Poisoner of Mayfair",
        "description": "Inspector Thomas arrives at an isolated Victorian country manor where a wealthy patriarch was found dead inside a locked study. Forensics reveal trace arsenic in vintage tea, a forged inheritance will, and contradictory alibis among seven suspicious guests."
    },
    {
        "genre": "Science Fiction",
        "title": "Neon Singularity 2145",
        "description": "A rogue synthetic artificial intelligence hacks orbital colony defense grids across the Jovian moon system. A cybernetic hacker with neural implants navigates virtual reality networks and deep-space wormhole gateways to abort an automated extinction beacon."
    },
    {
        "genre": "Fantasy",
        "title": "The Dragon Sorcerer's Prophecy",
        "description": "An orphaned elf apprentice discovers an ancient runic blade that marks him as heir to a fallen dynasty. Guided by an arcane grimoire, he embarks across enchanted forests to awaken slumbering fire-breathing dragons and vanquish the dark warlock."
    },
    {
        "genre": "Horror",
        "title": "The Nightmare of Blackwood Asylum",
        "description": "Paranormal researchers investigate an abandoned psychiatric hospital notorious for sadistic lobotomies. Trapped after dusk, they confront demonic poltergeists, weeping spectral apparitions, bleeding walls, and gruesome rituals in the sealed cellar."
    },
    {
        "genre": "Romance",
        "title": "Sunset and Champagne in Paris",
        "description": "An ambitious pastry chef and a brooding French vineyard owner enter a fake engagement to satisfy family expectations in Montmartre. But undeniable romantic chemistry, shared laughter in rain-swept cafes, and breathless kisses under the Eiffel Tower ignite true love."
    },
    {
        "genre": "Thriller",
        "title": "The 48-Hour Protocol",
        "description": "An undercover CIA operative races through Berlin to intercept a stolen radiological dirty bomb before a rogue mercenary syndicate activates the timer. With double agents inside intelligence headquarters, the ticking clock leaves zero margin for error."
    },
    {
        "genre": "Historical Fiction",
        "title": "Letters from the Somme",
        "description": "In 1916 during World War I, a young British officer endures trench warfare, poison gas attacks, and relentless artillery barrages along the Western Front, writing poignant letters home to his sweetheart in Yorkshire amidst the mud and slaughter."
    },
    {
        "genre": "Fiction",
        "title": "The Architecture of Regret",
        "description": "A quiet literary family drama following three estranged adult siblings who return to their childhood Midwestern home to settle their late mother's affairs, confronting generational grievances, suburban disillusionment, and fragile paths toward forgiveness."
    }
]

# ---------------------------------------------------------
# Load Trained Model and Artifacts
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_resources():
    model_path = os.path.join(SCRIPT_DIR, "model.joblib")
    vectorizer_path = os.path.join(SCRIPT_DIR, "vectorizer.joblib")
    metrics_path = os.path.join(SCRIPT_DIR, "evaluation_metrics.json")
    dataset_path = os.path.join(SCRIPT_DIR, "dataset.csv")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("Model artifacts not found! Please run `python train_model.py` to generate model.joblib and vectorizer.joblib.")
        st.stop()

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    df_dataset = pd.DataFrame()
    if os.path.exists(dataset_path):
        df_dataset = pd.read_csv(dataset_path)

    return model, vectorizer, metrics, df_dataset

model, vectorizer, metrics, df_dataset = load_resources()

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "book_title" not in st.session_state:
    st.session_state.book_title = PRESET_SAMPLES[0]["title"]
if "book_desc" not in st.session_state:
    st.session_state.book_desc = PRESET_SAMPLES[0]["description"]

def set_sample(genre_name: str):
    matched = [s for s in PRESET_SAMPLES if s["genre"].lower() == genre_name.lower()]
    if matched:
        st.session_state.book_title = matched[0]["title"]
        st.session_state.book_desc = matched[0]["description"]

def randomize_sample():
    choice = random.choice(PRESET_SAMPLES)
    st.session_state.book_title = choice["title"]
    st.session_state.book_desc = choice["description"]

def clear_inputs():
    st.session_state.book_title = ""
    st.session_state.book_desc = ""

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 Project Metadata")
    st.markdown(
        """
        - **Course:** Information Retrieval (CS402)
        - **Task:** Text Document Classification
        - **Model:** Multinomial Naive Bayes
        - **Smoothing:** Laplace Smoothing ($\\alpha = 1.0$)
        - **Feature Space:** Term Frequency (Bag-of-Words)
        - **Classes:** 8 Distinct Genres
        - **Dataset:** 200 Curated Books
        """
    )
    st.markdown("---")

    st.markdown("### 🎲 Preset Genre Quick Load")
    for s in PRESET_SAMPLES:
        genre_meta = GENRE_THEMES.get(s["genre"], {})
        if st.button(f"{genre_meta.get('icon', '📖')} {s['genre']}", key=f"side_btn_{s['genre']}", use_container_width=True):
            set_sample(s["genre"])
            st.rerun()

    st.markdown("---")
    if st.button("🎲 Pick Random Book", use_container_width=True):
        randomize_sample()
        st.rerun()

    if st.button("🧹 Clear Inputs", use_container_width=True):
        clear_inputs()
        st.rerun()

    st.markdown("---")
    st.caption("College IR Academic Prototype • Python 3.11 & Streamlit")

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-card">
    <div class="project-pill">🎓 Information Retrieval Laboratory Project</div>
    <div class="hero-title">Book Genre Classification Using Naive Bayes</div>
    <p class="hero-desc">
        An interactive machine learning prototype implementing <b>Multinomial Naive Bayes with Laplace Smoothing (α = 1.0)</b>. 
        Predicts book categories from titles and plot synopses, while exposing the exact <b>prior probabilities</b>, 
        <b>word conditional probabilities</b>, and <b>log-likelihood derivations</b> behind every prediction.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Interactive Prediction Results Modal Dialog
# ---------------------------------------------------------
@st.dialog("🏆 Genre Prediction & Naive Bayes IR Breakdown", width="large")
def show_prediction_modal(
    pred_genre,
    conf_score,
    s_classes,
    s_probas,
    match_toks,
    tok_counts,
    all_classes,
    log_priors,
    jll,
    genre_theme
):
    st.markdown(f"""
    <div class="winner-card" style="background: {genre_theme['gradient']}; margin-bottom: 18px;">
        <span class="confidence-badge">PREDICTED GENRE • CONFIDENCE: {conf_score:.1f}%</span>
        <div class="winner-genre-title">
            <span>{genre_theme['icon']}</span>
            <span>{pred_genre}</span>
        </div>
        <p style="margin: 0; font-size: 0.95rem; line-height: 1.5; color: rgba(255, 255, 255, 0.95);">
            {genre_theme['desc']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    pop_tab1, pop_tab2 = st.tabs(["📊 Probability Chart & Ranks", "🧮 Step-by-Step IR Mathematics"])

    with pop_tab1:
        p_c1, p_c2 = st.columns([1, 1.4])
        with p_c1:
            st.markdown("##### 🏅 Ranked Probability Breakdown")
            df_ranks = pd.DataFrame({
                "Rank": [f"#{i+1}" for i in range(len(s_classes))],
                "Genre": [f"{GENRE_THEMES.get(c, {}).get('icon', '')} {c}" for c in s_classes],
                "Probability": [f"{p*100:.2f}%" for p in s_probas]
            })
            st.dataframe(df_ranks, use_container_width=True, hide_index=True)

        with p_c2:
            st.markdown("##### 📈 Probability Distribution")
            df_pchart = pd.DataFrame({
                "Genre": s_classes,
                "Probability": [p * 100 for p in s_probas],
                "Color": [GENRE_THEMES.get(c, {}).get("color", "#3B82F6") for c in s_classes]
            })
            f_chart = px.bar(
                df_pchart,
                x="Probability",
                y="Genre",
                orientation="h",
                text=df_pchart["Probability"].apply(lambda x: f"{x:.1f}%"),
                color="Genre",
                color_discrete_map={c: GENRE_THEMES.get(c, {}).get("color", "#3B82F6") for c in s_classes}
            )
            f_chart.update_layout(
                yaxis=dict(autorange="reversed", title=""),
                xaxis=dict(title="Posterior Probability (%)", range=[0, max(100, conf_score + 12)]),
                showlegend=False,
                margin=dict(l=0, r=10, t=10, b=10),
                height=280
            )
            f_chart.update_traces(textposition="outside")
            st.plotly_chart(f_chart, use_container_width=True)

    with pop_tab2:
        st.markdown(f"##### 1. Matched Vocabulary Terms ({len(match_toks)} tokens)")
        tags_code = "".join([f"<span class='token-tag'>{w}<span class='token-count'>{cnt}</span></span>" for w, cnt in tok_counts.items()])
        st.markdown(tags_code, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### 2. Prior & Smoothing Parameters")
        st.write(f"- **Uniform Prior $P(C_k)$:** `12.5%` ($P = 0.125$) uniform across all 8 genres.")
        st.write(f"- **Laplace Smoothing:** $\\alpha = 1.0$ applied over $|V| = 2,121$ unique terms.")

        if match_toks:
            st.markdown("---")
            st.markdown("##### 3. Word Conditional Probabilities Comparison")
            top2 = s_classes[:2]
            g1_i = all_classes.index(top2[0])
            g2_i = all_classes.index(top2[1])
            comp_rows = []
            for w in list(tok_counts.keys())[:8]:
                w_id = vocab[w]
                pw1 = np.exp(model.feature_log_prob_[g1_i, w_id])
                pw2 = np.exp(model.feature_log_prob_[g2_i, w_id])
                comp_rows.append({
                    "Word": w,
                    f"P(w | {top2[0]})": f"{pw1:.5f}",
                    f"P(w | {top2[1]})": f"{pw2:.5f}",
                    "Favors": top2[0] if pw1 > pw2 else top2[1]
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("##### 4. Joint Log-Likelihood Score")
        win_score = jll[all_classes.index(pred_genre)]
        st.success(f"🎯 Genre **{pred_genre}** achieved maximum joint log-likelihood score **{win_score:.2f}** ({conf_score:.2f}%).")


# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_predict, tab_evaluation, tab_dataset, tab_viva = st.tabs([
    "🔮 Predict Genre & IR Calculations",
    "📊 Model Evaluation & IR Metrics",
    "📁 Dataset Explorer",
    "📖 Naive Bayes Theory & Viva Guide"
])

# =========================================================
# TAB 1: PREDICT GENRE & IR MATHEMATICAL BREAKDOWN
# =========================================================
with tab_predict:
    st.markdown("#### ⚡ Quick Genre Loaders (Click to Test)")
    
    # 8 Interactive Pill Buttons
    chip_cols = st.columns(8)
    genres_list = list(GENRE_THEMES.keys())
    for i, g in enumerate(genres_list):
        meta = GENRE_THEMES[g]
        with chip_cols[i]:
            if st.button(f"{meta['icon']} {g}", key=f"chip_{g}", use_container_width=True):
                set_sample(g)
                st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    input_col, info_col = st.columns([2.3, 1], gap="large")

    with input_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 📝 Book Query Input")
        
        book_title_val = st.text_input(
            "Book Title",
            value=st.session_state.book_title,
            placeholder="e.g., The Hound of the Baskervilles",
            key="widget_book_title"
        )
        
        book_desc_val = st.text_area(
            "Book Description / Plot Synopsis",
            value=st.session_state.book_desc,
            placeholder="Type or paste a plot summary, back-cover blurb, or book synopsis...",
            height=140,
            key="widget_book_desc"
        )

        # Real-time Stats Strip
        full_query_text = f"{book_title_val.strip()} {book_desc_val.strip()}".strip()
        word_count = len(full_query_text.split()) if full_query_text else 0
        char_count = len(full_query_text)

        # Token overlap check with vectorizer vocabulary
        analyzer = vectorizer.build_analyzer()
        extracted_tokens = analyzer(full_query_text) if full_query_text else []
        vocab = vectorizer.vocabulary_
        matched_tokens = [w for w in extracted_tokens if w in vocab]
        oov_tokens = [w for w in extracted_tokens if w not in vocab]
        token_series = pd.Series(matched_tokens).value_counts()
        token_counts = token_series.to_dict()
        
        st.markdown(
            f"""
            <div style="display: flex; gap: 15px; margin-top: 6px; margin-bottom: 14px; font-size: 0.82rem; color: #64748B;">
                <span>📏 <b>{char_count}</b> Characters</span>
                <span>📝 <b>{word_count}</b> Words</span>
                <span>🎯 <b>{len(matched_tokens)}</b> Vocabulary Matches</span>
                <span>🚫 <b>{len(oov_tokens)}</b> OOV / Stopwords</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Pre-compute prediction if text exists
        if full_query_text:
            text_vec = vectorizer.transform([full_query_text])
            predicted_genre = model.predict(text_vec)[0]
            probas = model.predict_proba(text_vec)[0]
            classes = list(model.classes_)
            joint_log_lik = model._joint_log_likelihood(text_vec)[0]
            class_log_priors = model.class_log_prior_

            sorted_indices = np.argsort(probas)[::-1]
            sorted_classes = [classes[i] for i in sorted_indices]
            sorted_probas = [probas[i] for i in sorted_indices]
            top_confidence = sorted_probas[0] * 100
            theme = GENRE_THEMES.get(predicted_genre, GENRE_THEMES["Fiction"])
        else:
            predicted_genre = "Unknown"
            top_confidence = 0.0
            sorted_classes = list(model.classes_)
            sorted_probas = [0.125] * len(sorted_classes)
            classes = list(model.classes_)
            joint_log_lik = [0.0] * len(classes)
            class_log_priors = [0.0] * len(classes)
            probas = [0.125] * len(classes)
            theme = GENRE_THEMES["Fiction"]

        b_col1, b_col2, b_col3, b_col4 = st.columns([1.8, 1.4, 1.3, 1.1])
        with b_col1:
            btn_predict = st.button("🚀 Predict & Pop Up", type="primary", use_container_width=True)
        with b_col2:
            btn_popup_manual = st.button("✨ Pop Up Window", use_container_width=True)
        with b_col3:
            if st.button("🎲 Sample Book", use_container_width=True):
                randomize_sample()
                st.rerun()
        with b_col4:
            if st.button("🧹 Clear", use_container_width=True):
                clear_inputs()
                st.rerun()

        # Handle Popup Trigger
        if (btn_predict or btn_popup_manual) and full_query_text:
            st.toast(f"🎯 Predicted: {predicted_genre} ({top_confidence:.1f}%)", icon=theme['icon'])
            st.balloons()
            show_prediction_modal(
                predicted_genre,
                top_confidence,
                sorted_classes,
                sorted_probas,
                matched_tokens,
                token_counts,
                classes,
                class_log_priors,
                joint_log_lik,
                theme
            )
        elif (btn_predict or btn_popup_manual) and not full_query_text:
            st.warning("⚠️ Please enter a book title or description first!")

        # Prominent Result Ribbon right inside the input card
        if full_query_text:
            st.markdown(f"""
            <div style="background: {theme['bg_light']}; border: 1.5px solid {theme['color']}; border-radius: 12px; padding: 12px 18px; margin-top: 14px; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="font-size: 1.15rem; font-weight: 800; color: {theme['color']};">{theme['icon']} Predicted: {predicted_genre}</span>
                    <span style="background: rgba(0,0,0,0.06); padding: 3px 8px; border-radius: 8px; font-weight: 700; color: #334155; margin-left: 8px; font-size: 0.88rem;">{top_confidence:.1f}% Confidence</span>
                </div>
                <span style="color: #64748B; font-size: 0.8rem; font-weight: 600;">Click "🚀 Predict & Pop Up" to open modal dialog</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with info_col:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown("##### 💡 Information Retrieval Context")
        st.markdown(
            """
            In an IR search engine or digital library:
            1. **Query Document $D$:** The book description is represented as a bag of word frequencies $f(w)$.
            2. **Prior $P(C_k)$:** Baseline probability of encountering genre $C_k$.
            3. **Likelihood $P(D|C_k)$:** Product of conditional word probabilities $\prod P(w_i | C_k)$.
            4. **Posterior $P(C_k|D)$:** The normalized confidence that this book belongs to genre $C_k$.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Classification Computation & Results
    # ---------------------------------------------------------
    if full_query_text:
        text_vec = vectorizer.transform([full_query_text])
        predicted_genre = model.predict(text_vec)[0]
        probas = model.predict_proba(text_vec)[0]
        classes = list(model.classes_)

        # Native Scikit-Learn Joint Log Likelihood (Zero array method issues!)
        joint_log_lik = model._joint_log_likelihood(text_vec)[0]
        class_log_priors = model.class_log_prior_

        # Sort classes descending by probability
        sorted_indices = np.argsort(probas)[::-1]
        sorted_classes = [classes[i] for i in sorted_indices]
        sorted_probas = [probas[i] for i in sorted_indices]
        top_confidence = sorted_probas[0] * 100

        theme = GENRE_THEMES.get(predicted_genre, GENRE_THEMES["Fiction"])

        st.markdown("---")
        st.markdown("### 🏆 Prediction Results")

        res_col_left, res_col_right = st.columns([1.1, 1.5], gap="large")

        with res_col_left:
            # Winner Showcase Card
            st.markdown(f"""
            <div class="winner-card" style="background: {theme['gradient']};">
                <span class="confidence-badge">PREDICTED GENRE • CONFIDENCE: {top_confidence:.1f}%</span>
                <div class="winner-genre-title">
                    <span>{theme['icon']}</span>
                    <span>{predicted_genre}</span>
                </div>
                <p style="margin: 0; font-size: 0.95rem; line-height: 1.5; color: rgba(255, 255, 255, 0.92);">
                    {theme['desc']}
                </p>
                <div style="margin-top: 16px; padding-top: 14px; border-top: 1px solid rgba(255,255,255,0.2); display: flex; justify-content: space-between;">
                    <span>Model Confidence Rating:</span>
                    <b>{'⭐⭐⭐ High Certainty' if top_confidence > 50 else '⭐⭐ Moderate Certainty'}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Top Ranked Table
            st.markdown("##### 🏅 Ranked Probability Breakdown")
            ranked_df = pd.DataFrame({
                "Rank": [f"#{i+1}" for i in range(len(sorted_classes))],
                "Genre": [f"{GENRE_THEMES.get(c, {}).get('icon', '')} {c}" for c in sorted_classes],
                "Probability": [f"{p*100:.2f}%" for p in sorted_probas]
            })
            st.dataframe(ranked_df, use_container_width=True, hide_index=True)

        with res_col_right:
            st.markdown("##### 📊 Genre Probability Scores (%)")
            chart_df = pd.DataFrame({
                "Genre": sorted_classes,
                "Probability": [p * 100 for p in sorted_probas],
                "Color": [GENRE_THEMES.get(c, {}).get("color", "#3B82F6") for c in sorted_classes]
            })

            fig = px.bar(
                chart_df,
                x="Probability",
                y="Genre",
                orientation="h",
                text=chart_df["Probability"].apply(lambda x: f"{x:.1f}%"),
                color="Genre",
                color_discrete_map={c: GENRE_THEMES.get(c, {}).get("color", "#3B82F6") for c in sorted_classes}
            )
            fig.update_layout(
                yaxis=dict(autorange="reversed", title=""),
                xaxis=dict(title="Posterior Probability (%)", range=[0, max(100, top_confidence + 12)]),
                showlegend=False,
                margin=dict(l=0, r=20, t=10, b=10),
                height=340,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            fig.update_traces(
                textposition="outside",
                marker=dict(line=dict(width=0))
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------------------------
        # SECTION 4: SHOW NAIVE BAYES CALCULATIONS
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🧮 How the Naive Bayes Prediction Was Calculated")
        st.caption("Step-by-step mathematical breakdown using the actual fitted parameters from the Multinomial Naive Bayes model.")

        with st.expander("🔍 Click to Inspect Full IR Mathematical Derivation & Parameter Breakdown", expanded=True):
            
            # 1. Term Tokenization
            st.markdown("#### 1. Term Tokenization & Stopword Filtering")
            st.markdown(
                f"The text query was preprocessed into tokens using CountVectorizer (lowercased, alphanumeric tokens, English stop-words removed).<br>"
                f"- **Recognized Vocabulary Words:** `{len(matched_tokens)}` tokens<br>"
                f"- **Out-of-Vocabulary (OOV) / Removed Stopwords:** `{len(oov_tokens)}` words",
                unsafe_allow_html=True
            )

            token_series = pd.Series(matched_tokens).value_counts()
            token_counts = token_series.to_dict()

            if matched_tokens:
                st.markdown("**Matched Vocabulary Keywords (and counts in query):**")
                tags_markup = "".join([f"<span class='token-tag'>{word}<span class='token-count'>{cnt}</span></span>" for word, cnt in token_counts.items()])
                st.markdown(tags_markup, unsafe_allow_html=True)
            else:
                st.warning("⚠️ None of the words in the description occurred in the training vocabulary. The prediction defaults to the prior probability distribution.")

            st.markdown("---")

            # 2. Prior Probabilities P(C_k)
            st.markdown("#### 2. Class Prior Probabilities: $P(C_k)$")
            st.markdown(
                """
                In Information Retrieval classification, the **prior probability** $P(C_k)$ is the empirical proportion 
                of training documents belonging to genre $C_k$ before reading any words:
                $$P(C_k) = \\frac{N_{C_k}}{N_{\\text{total}}}$$
                """
            )

            priors = {cls: np.exp(model.class_log_prior_[i]) for i, cls in enumerate(classes)}
            prior_df = pd.DataFrame({
                "Genre": [f"{GENRE_THEMES.get(c, {}).get('icon', '')} {c}" for c in classes],
                "Training Samples (N_c)": [int(model.class_count_[i]) for i in range(len(classes))],
                "Prior Probability P(C_k)": [f"{priors[c]:.4f} ({priors[c]*100:.1f}%)" for c in classes],
                "Log Prior log P(C_k)": [f"{model.class_log_prior_[i]:.4f}" for i in range(len(classes))]
            })
            st.dataframe(prior_df, use_container_width=True, hide_index=True)
            st.info("💡 **Uniform Prior Note:** Because our curated dataset contains exactly 20 training books per genre (160 total), every genre starts with an identical prior of **12.5%** ($P = 0.125$).")

            st.markdown("---")

            # 3. Laplace Smoothing Explanation
            st.markdown("#### 3. Laplace Smoothing ($\alpha = 1.0$) Explanation")
            st.markdown(
                r"""
                **The Zero-Frequency Problem in IR:**
                If a word $w$ appears in the test book but never appeared in training books for genre $C_k$, the raw empirical frequency is zero:
                $$P(w | C_k) = \frac{\text{count}(w, C_k)}{\sum_{w'} \text{count}(w', C_k)} = 0$$
                Because Naive Bayes multiplies all word probabilities together ($\prod P(w_i | C_k)$), **a single zero would cancel out the entire product to zero**, 
                even if every other word in the synopsis screamed that genre!

                **Laplace (Add-1) Smoothing Solution:**
                We add $\alpha = 1.0$ to the numerator, and add $\alpha \cdot |V|$ (where $|V|$ is the total unique vocabulary size) to the denominator:
                $$\hat{P}(w_i | C_k) = \frac{\text{count}(w_i, C_k) + \alpha}{\sum_{w' \in V} \text{count}(w', C_k) + \alpha \cdot |V|}$$
                """
            )
            v_size = len(vectorizer.vocabulary_)
            st.markdown(f"- **Fitted Vocabulary Size $|V|$:** `{v_size}` unique terms")
            st.markdown(f"- **Smoothing Parameter $\\alpha$:** `1.0` (Standard Laplace)")

            st.markdown("---")

            # 4. Conditional Probabilities Table
            st.markdown("#### 4. Word Conditional Probabilities: $P(w_i | C_k)$")
            st.markdown(
                "Below are the actual training counts and fitted conditional probabilities for query words, "
                "comparing the **top 2 predicted genres**:"
            )

            if matched_tokens:
                top_2_genres = sorted_classes[:2]
                g1_idx = classes.index(top_2_genres[0])
                g2_idx = classes.index(top_2_genres[1])

                sample_words = list(token_counts.keys())[:15]
                word_comp_rows = []

                for w in sample_words:
                    w_id = vocab[w]
                    cnt_g1 = model.feature_count_[g1_idx, w_id]
                    cnt_g2 = model.feature_count_[g2_idx, w_id]
                    p_g1 = np.exp(model.feature_log_prob_[g1_idx, w_id])
                    p_g2 = np.exp(model.feature_log_prob_[g2_idx, w_id])

                    ratio = p_g1 / p_g2 if p_g2 > 0 else 1.0
                    favors = top_2_genres[0] if ratio > 1.08 else (top_2_genres[1] if ratio < 0.92 else "Neutral")

                    word_comp_rows.append({
                        "Query Word": w,
                        "Query Count": token_counts[w],
                        f"Count in {top_2_genres[0]}": int(cnt_g1),
                        f"P(w | {top_2_genres[0]})": f"{p_g1:.5f}",
                        f"Count in {top_2_genres[1]}": int(cnt_g2),
                        f"P(w | {top_2_genres[1]})": f"{p_g2:.5f}",
                        "Favors": favors
                    })

                st.dataframe(pd.DataFrame(word_comp_rows), use_container_width=True, hide_index=True)

            st.markdown("---")

            # 5. Log-Likelihood & Posterior Normalization
            st.markdown("#### 5. Log-Likelihood & Posterior Probability Computation")
            st.markdown(
                r"""
                To prevent arithmetic floating-point underflow when multiplying dozens of small decimal probabilities, 
                Information Retrieval systems sum in log-space:
                $$\log P(C_k | D) \propto \log P(C_k) + \sum_{i=1}^m f(w_i) \cdot \log P(w_i | C_k)$$
                
                The final normalized posterior probability $P(C_k | D)$ is obtained via the **Log-Sum-Exp (Softmax)** normalization:
                $$P(C_k | D) = \frac{\exp(\log P(C_k | D))}{\sum_j \exp(\log P(C_j | D))}$$
                """
            )

            log_calc_df = pd.DataFrame({
                "Genre": [f"{GENRE_THEMES.get(c, {}).get('icon', '')} {c}" for c in classes],
                "Log Prior log P(C_k)": [f"{class_log_priors[i]:.2f}" for i in range(len(classes))],
                "Sum of Word Log-Likelihoods": [f"{(joint_log_lik[i] - class_log_priors[i]):.2f}" for i in range(len(classes))],
                "Total Unnormalized Log-Score": [f"{joint_log_lik[i]:.2f}" for i in range(len(classes))],
                "Posterior P(C_k | D)": [f"{probas[i]*100:.2f}%" for i in range(len(classes))]
            })
            log_calc_df["Sort_Key"] = probas
            log_calc_df = log_calc_df.sort_values(by="Sort_Key", ascending=False).drop(columns=["Sort_Key"])

            st.dataframe(log_calc_df, use_container_width=True, hide_index=True)

            winner_log_score = joint_log_lik[classes.index(predicted_genre)]
            st.success(
                f"🎯 **Final Decision Rule:** Genre **{predicted_genre}** maximizes the joint log-likelihood "
                f"with a score of **{winner_log_score:.2f}**, yielding the highest posterior probability of **{top_confidence:.2f}%**."
            )

    else:
        st.info("👋 Enter a book title and description above or click one of the quick genre sample buttons to see the live classification!")

# =========================================================
# TAB 2: MODEL EVALUATION & METRICS (Section 5)
# =========================================================
with tab_evaluation:
    st.markdown("### 📊 Empirical Model Evaluation")
    st.caption(
        "Evaluated on a strictly independent held-out test dataset (20% = 40 samples, 5 per genre) "
        "never seen during training to prevent data leakage."
    )

    if metrics:
        # 4 KPI Glass Cards
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color: #2563EB;">{metrics.get('accuracy', 0)*100:.1f}%</div>
                <div class="kpi-label">🎯 Overall Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color: #7C3AED;">{metrics.get('precision_macro', 0)*100:.1f}%</div>
                <div class="kpi-label">🔍 Macro Precision</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color: #059669;">{metrics.get('recall_macro', 0)*100:.1f}%</div>
                <div class="kpi-label">📈 Macro Recall</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_col4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color: #D97706;">{metrics.get('f1_macro', 0)*100:.1f}%</div>
                <div class="kpi-label">⚖️ Macro F1-Score</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        ev_left, ev_right = st.columns([1.1, 1], gap="large")

        with ev_left:
            st.markdown("##### 🔲 Confusion Matrix Heatmap")
            cm = np.array(metrics.get("confusion_matrix", []))
            lbls = metrics.get("classes", list(GENRE_THEMES.keys()))

            fig_cm = px.imshow(
                cm,
                x=lbls,
                y=lbls,
                color_continuous_scale="Blues",
                text_auto=True,
                labels=dict(x="Predicted Genre", y="True Ground Truth Genre", color="Count")
            )
            fig_cm.update_layout(
                xaxis=dict(tickangle=-35),
                height=430,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with ev_right:
            st.markdown("##### 📋 Per-Genre Classification Report")
            clf_rep = metrics.get("classification_report", {})
            report_rows = []
            for cls_name in lbls:
                if cls_name in clf_rep:
                    row = clf_rep[cls_name]
                    report_rows.append({
                        "Genre": f"{GENRE_THEMES.get(cls_name, {}).get('icon', '')} {cls_name}",
                        "Precision": f"{row['precision']*100:.1f}%",
                        "Recall": f"{row['recall']*100:.1f}%",
                        "F1-Score": f"{row['f1-score']*100:.1f}%",
                        "Test Support": int(row['support'])
                    })
            st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

            st.markdown("""
            <div class="glass-card" style="padding: 14px; margin-top: 14px;">
                <b style="color: #1E293B;">🛡️ Evaluation Integrity Notes:</b>
                <ul style="margin: 6px 0 0 0; padding-left: 20px; font-size: 0.88rem; color: #475569;">
                    <li><b>Stratified 80/20 Split:</b> 160 training books (20/genre) and 40 test books (5/genre).</li>
                    <li><b>Zero Data Leakage:</b> CountVectorizer was fitted strictly on train data and applied to test data.</li>
                    <li><b>Academic Notice:</b> Synthetic/curated demonstration dataset used to illustrate IR text classification mechanics.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🧪 Interactive Held-Out Test Set Inspector")
        st.markdown("Select any real test document from the held-out partition to verify predictions against ground truth:")

        test_samples = metrics.get("test_samples", [])
        if test_samples:
            sample_choices = [
                f"[{'✅ Correct' if s['is_correct'] else '❌ Error'}] {s['title']} (True: {s['genre']} | Pred: {s['predicted_genre']})"
                for s in test_samples
            ]
            chosen_idx = st.selectbox("Choose a test book to inspect:", range(len(test_samples)), format_func=lambda i: sample_choices[i])
            chosen_obj = test_samples[chosen_idx]

            tc1, tc2 = st.columns([2.5, 1])
            with tc1:
                st.markdown(f"**Title:** *{chosen_obj['title']}*")
                st.markdown(f"**Plot Synopsis:** {chosen_obj['description']}")
            with tc2:
                st.markdown(f"**True Label:** `{chosen_obj['genre']}`")
                st.markdown(f"**Predicted Label:** `{chosen_obj['predicted_genre']}`")
                if chosen_obj["is_correct"]:
                    st.success("✅ Prediction Matched True Genre")
                else:
                    st.error("❌ Prediction Misclassified")
                if st.button("📥 Load into Predictor Tab"):
                    st.session_state.book_title = chosen_obj["title"]
                    st.session_state.book_desc = chosen_obj["description"]
                    st.rerun()

    else:
        st.warning("No evaluation metrics found. Please run `train_model.py`.")

# =========================================================
# TAB 3: DATASET EXPLORER (Section 3)
# =========================================================
with tab_dataset:
    st.markdown("### 📁 Demonstration Dataset Explorer")
    st.markdown(
        """
        > [!NOTE]
        > **Demonstration Dataset Notice:**
        > This corpus contains **200 verified book descriptions** (25 per genre across 8 classes).
        > It is curated specifically for this college Information Retrieval assignment to demonstrate feature extraction, 
        > smoothing behavior, and genre classification.
        """
    )

    if not df_dataset.empty:
        filter_col1, filter_col2 = st.columns([1, 2])
        with filter_col1:
            genres_filter = st.multiselect(
                "Filter Genres:",
                options=list(df_dataset["genre"].unique()),
                default=list(df_dataset["genre"].unique())
            )
        with filter_col2:
            search_term = st.text_input("🔍 Search keyword in title or description:", "")

        view_df = df_dataset[df_dataset["genre"].isin(genres_filter)]
        if search_term:
            view_df = view_df[
                view_df["title"].str.contains(search_term, case=False, na=False) |
                view_df["description"].str.contains(search_term, case=False, na=False)
            ]

        st.markdown(f"Showing **{len(view_df)}** of **{len(df_dataset)}** books:")
        st.dataframe(view_df, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📊 Dataset Distributions")
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            st.markdown("##### Book Count per Genre")
            counts = df_dataset["genre"].value_counts().reset_index()
            counts.columns = ["Genre", "Count"]
            fig_bar = px.bar(
                counts,
                x="Genre",
                y="Count",
                color="Genre",
                color_discrete_map={g: GENRE_THEMES.get(g, {}).get("color", "#3B82F6") for g in counts["Genre"]}
            )
            fig_bar.update_layout(showlegend=False, height=290, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

        with d_c2:
            st.markdown("##### Description Word Count Distribution")
            df_dataset["words"] = df_dataset["description"].apply(lambda x: len(str(x).split()))
            fig_box = px.box(
                df_dataset,
                x="genre",
                y="words",
                color="genre",
                color_discrete_map={g: GENRE_THEMES.get(g, {}).get("color", "#3B82F6") for g in df_dataset["genre"].unique()}
            )
            fig_box.update_layout(showlegend=False, height=290, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Words in Description")
            st.plotly_chart(fig_box, use_container_width=True)

# =========================================================
# TAB 4: THEORY & VIVA GUIDE
# =========================================================
with tab_viva:
    st.markdown("### 📖 Naive Bayes in Information Retrieval & Viva Q&A Guide")
    st.markdown(
        r"""
        #### 🎓 Frequently Asked Questions in Course Viva / Lab Presentations

        ##### 1. What is the Bag-of-Words (BoW) model?
        The Bag-of-Words model represents a text document as an unordered collection of term frequencies. Word order and grammatical structure are ignored, preserving only the occurrences of each vocabulary word. In our project, `CountVectorizer` converts book descriptions into term-frequency vectors after stripping English stop-words.

        ##### 2. Why is Naive Bayes called "Naive"?
        It is called "naive" because it makes the strong (and linguistically unrealistic) assumption that **all words in a document are conditionally independent of each other**, given the document class:
        $$P(w_1, w_2, \dots, w_m \mid C_k) = \prod_{i=1}^m P(w_i \mid C_k)$$
        In natural language, words like "artificial" and "intelligence" are heavily dependent. However, in Information Retrieval, Naive Bayes performs surprisingly well because classification depends on which genre has the highest score (relative ranking) rather than exact probability calibration.

        ##### 3. Why is Laplace Smoothing ($\alpha = 1.0$) essential in Information Retrieval?
        If an input document contains a word $w$ that never appeared in genre $C_k$ during training, the maximum likelihood estimate is:
        $$P(w \mid C_k) = \frac{\text{count}(w, C_k)}{N_{C_k}} = 0$$
        Because the Naive Bayes likelihood multiplies all term probabilities together, **a single zero will zero out the entire product**, preventing the model from ever predicting that genre regardless of all other words. Laplace smoothing adds $\alpha = 1$ to the numerator and $\alpha \cdot |V|$ to the denominator:
        $$P(w_i \mid C_k) = \frac{\text{count}(w_i, C_k) + 1}{\sum_{w'} \text{count}(w', C_k) + |V|}$$

        ##### 4. Why do we compute probabilities in Log-Space?
        Multiplying dozens of fractional probabilities ($0.001 \times 0.0005 \times \dots$) quickly causes **arithmetic floating-point underflow** in computer systems. By converting to log-space, multiplication turns into addition:
        $$\log P(C_k \mid D) \propto \log P(C_k) + \sum_{i=1}^m \log P(w_i \mid C_k)$$

        ##### 5. How does Naive Bayes compare with TF-IDF and Neural Models?
        - **Naive Bayes:** Fast training ($\mathcal{O}(|D| \cdot |C|)$), highly interpretable, lightweight, ideal for cold-start indexing.
        - **TF-IDF + SVM/Logistic Regression:** Weights terms by inverse document frequency, penalizing universally frequent words.
        - **Transformers (BERT, RoBERTa):** Captures semantic context and word order, but requires heavy compute and extensive training corpora.
        """
    )
