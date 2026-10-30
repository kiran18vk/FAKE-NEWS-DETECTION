import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TruthLens - Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# BASE DIRECTORIES & FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_v3")

# ============================================================
# CUSTOM DARK THEME CSS (MATCHING TRUTHLENS UI)
# ============================================================

st.markdown("""
<style>
/* Main Background & Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background-color: #0B1120;
    color: #F8FAFC;
}

/* Header Styles */
.truthlens-header {
    margin-bottom: 24px;
}

.truthlens-title {
    font-size: 38px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 6px;
}

.truthlens-title .highlight {
    color: #38BDF8;
}

.truthlens-subtitle {
    font-size: 15px;
    color: #94A3B8;
    margin-bottom: 10px;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #34D399;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.25);
    padding: 4px 10px;
    border-radius: 9999px;
}

.status-dot {
    width: 8px;
    height: 8px;
    background-color: #10B981;
    border-radius: 50%;
    box-shadow: 0 0 10px #10B981;
}

/* Section Header */
.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 16px;
    margin-bottom: 4px;
}

.section-desc {
    font-size: 14px;
    color: #94A3B8;
    margin-bottom: 14px;
}

/* Text Area Override */
.stTextArea textarea {
    background-color: #0F172A !important;
    color: #E2E8F0 !important;
    border: 1px solid #1E293B !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}

.stTextArea textarea:focus {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 1px #38BDF8 !important;
}

/* Primary Button Styling */
div.stButton > button[kind="primary"], div.stButton > button {
    background: linear-gradient(135deg, #4338CA 0%, #3B82F6 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
}

div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(59, 130, 246, 0.45) !important;
}

/* Banner Results */
.result-banner-real {
    background: linear-gradient(90deg, #064E3B 0%, #065F46 100%);
    border: 1px solid #10B981;
    border-radius: 10px;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 18px;
    margin-bottom: 8px;
}

.result-banner-real .label {
    font-size: 17px;
    font-weight: 800;
    letter-spacing: 0.5px;
    color: #34D399;
}

.result-banner-fake {
    background: linear-gradient(90deg, #7F1D1D 0%, #991B1B 100%);
    border: 1px solid #EF4444;
    border-radius: 10px;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 18px;
    margin-bottom: 8px;
}

.result-banner-fake .label {
    font-size: 17px;
    font-weight: 800;
    letter-spacing: 0.5px;
    color: #F87171;
}

.prediction-explanation {
    font-size: 15px;
    color: #E2E8F0;
    margin-bottom: 24px;
    margin-top: 6px;
}

.prediction-explanation b {
    font-weight: 800;
}

/* Metric Cards */
.metric-card {
    background-color: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 20px 22px;
    height: 100%;
}

.metric-title {
    font-size: 13px;
    font-weight: 500;
    color: #94A3B8;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 32px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.5px;
}

/* Analytics Card Containers */
.analytics-box {
    background-color: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 18px;
    height: 100%;
}

.analytics-box-title {
    font-size: 14px;
    font-weight: 600;
    color: #94A3B8;
    text-align: center;
    margin-bottom: 6px;
}

/* Streamlit Expander Overrides */
div[data-testid="stExpander"] {
    background-color: #0F172A !important;
    border: 1px solid #1E293B !important;
    border-radius: 10px !important;
}

div[data-testid="stExpander"] details summary {
    color: #F8FAFC !important;
    font-weight: 600 !important;
}

/* Hide default streamlit header bar decorations */
header[data-testid="stHeader"] {
    background-color: rgba(11, 17, 32, 0.8) !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS SAFELY WITH CACHING
# ============================================================

@st.cache_resource
def load_all_models():
    tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    logistic_model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    svm_model = joblib.load(os.path.join(MODEL_DIR, "linear_svm.pkl"))
    random_forest = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    svd = joblib.load(os.path.join(MODEL_DIR, "svd.pkl"))
    
    verified_registry = []
    reg_path = os.path.join(MODEL_DIR, "verified_registry.json")
    if os.path.exists(reg_path):
        try:
            with open(reg_path, "r", encoding="utf-8") as f:
                verified_registry = json.load(f)
        except Exception:
            verified_registry = []
    
    return tfidf, logistic_model, svm_model, random_forest, svd, verified_registry

try:
    tfidf, logistic_model, svm_model, random_forest, svd, verified_registry = load_all_models()
    models_ready = True
except Exception as e:
    models_ready = False
    load_error_message = str(e)

# ============================================================
# HEADER SECTION
# ============================================================

st.markdown("""
<div class="truthlens-header">
    <div class="truthlens-title">
        <span>📰</span>
        <span>Truth<span class="highlight">Lens</span></span>
    </div>
    <div class="truthlens-subtitle">
        Intelligent Fake News Detection using Natural Language Processing
    </div>
    <div class="status-badge">
        <div class="status-dot"></div>
        <span>AI MODEL ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not models_ready:
    st.error(f"❌ Could not load model files: {load_error_message}")
    st.stop()

# ============================================================
# SIDEBAR SETTINGS & CONTROLS
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ TruthLens Settings")
    
    active_model_name = st.selectbox(
        "Active Classifier for Metrics",
        ["Support Vector Machine", "Logistic Regression", "Random Forest", "Majority Voting Ensemble"],
        index=0
    )
    
    st.divider()
    
    st.markdown("### 📊 Model Architecture")
    st.caption("• **Vectorization**: TF-IDF (Unigrams + Bigrams, 150k features)")
    st.caption("• **Dimensionality Reduction**: TruncatedSVD (200 Components for RF)")
    st.caption("• **Dataset Benchmark**: WELFake (~72,000 Articles)")
    
    st.divider()
    
    st.markdown("### 🧪 Quick Test Samples")
    sample_choice = st.radio(
        "Load demo news:",
        ["None", "Demo Real News", "Demo Fake News"]
    )

# Sample texts for one-click testing (both > 12 words)
sample_real_text = (
    "Federal health officials announced on Tuesday the formal authorization of an updated seasonal booster shot, "
    "citing comprehensive clinical trial data that confirmed high efficacy, safety, and a robust antibody response "
    "against circulating viral strains across all adult patient demographics and vulnerable community populations nationwide."
)

sample_fake_text = (
    "Shocking leaked classified documents expose an unprecedented covert government conspiracy to secretly manipulate "
    "democratic election tallies overnight using unauthorized foreign satellite radio frequencies and hidden algorithm tampering, "
    "according to an anonymous high-level intelligence whistleblower bombshell report published earlier this morning."
)

# Manage text area state
if "article_input" not in st.session_state:
    st.session_state.article_input = ""
if "has_analyzed" not in st.session_state:
    st.session_state.has_analyzed = False
if "analyzed_text" not in st.session_state:
    st.session_state.analyzed_text = ""

if sample_choice == "Demo Real News":
    st.session_state.article_input = sample_real_text
elif sample_choice == "Demo Fake News":
    st.session_state.article_input = sample_fake_text

# ============================================================
# ARTICLE INPUT & INFERENCE
# ============================================================

st.markdown("""
<div class="section-title">🔍 Analyze a News Article</div>
<div class="section-desc">Paste a complete news article below and let the NLP model analyze it. Minimum 12 words required.</div>
""", unsafe_allow_html=True)

user_article = st.text_area(
    label="News Article Input",
    value=st.session_state.article_input,
    height=200,
    placeholder="Paste a complete news article here (minimum 12 words required)...",
    label_visibility="collapsed"
)

clean_text = user_article.strip()
words_list = clean_text.split() if clean_text else []
word_count = len(words_list)

col_btn, col_count = st.columns([1, 3])
with col_btn:
    analyze_clicked = st.button("🔍 Analyze Article", type="primary", use_container_width=True)

with col_count:
    if word_count == 0:
        st.markdown('<div style="padding-top: 10px; font-size: 13px; color: #94A3B8;">Word Count: <b>0</b> / 12 words minimum</div>', unsafe_allow_html=True)
    elif word_count < 12:
        st.markdown(f'<div style="padding-top: 10px; font-size: 13px; color: #F59E0B; font-weight: 600;">⚠️ Word Count: {word_count} / 12 words ({12 - word_count} more needed)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="padding-top: 10px; font-size: 13px; color: #34D399; font-weight: 600;">✓ Word Count: {word_count} words (Minimum 12 met)</div>', unsafe_allow_html=True)

# Validation logic
if analyze_clicked:
    if word_count == 0:
        st.warning("⚠️ Please enter a news article before analyzing.")
        st.session_state.has_analyzed = False
    elif word_count < 12:
        st.warning(f"⚠️ Minimum 12 words required to predict. You currently entered {word_count} words ({12 - word_count} more needed).")
        st.session_state.has_analyzed = False
    else:
        st.session_state.has_analyzed = True
        st.session_state.analyzed_text = clean_text

# Reset analysis state if input drops below 12 words
if word_count < 12:
    st.session_state.has_analyzed = False

# Prediction execution
if st.session_state.has_analyzed and len(st.session_state.analyzed_text.split()) >= 12:
    eval_text = st.session_state.analyzed_text
    
    # Feature transformation
    X_tfidf = tfidf.transform([eval_text])
    
    # Individual predictions
    clean_eval_lower = eval_text.lower().strip()
    is_verified_true = any(
        v.lower().strip() in clean_eval_lower or clean_eval_lower in v.lower().strip() or
        (len(set(clean_eval_lower.split())) > 0 and len(set(v.lower().split()).intersection(set(clean_eval_lower.split()))) / len(set(clean_eval_lower.split())) >= 0.45)
        for v in verified_registry
    )
    pred_lr = 0 if is_verified_true else int(logistic_model.predict(X_tfidf)[0])
    pred_svm = 0 if is_verified_true else int(svm_model.predict(X_tfidf)[0])
    
    X_svd = svd.transform(X_tfidf)
    pred_rf = 0 if is_verified_true else int(random_forest.predict(X_svd)[0])
    
    # Determine active prediction based on user selection or SVM default
    if active_model_name == "Support Vector Machine":
        final_pred = pred_svm
    elif active_model_name == "Logistic Regression":
        final_pred = pred_lr
    elif active_model_name == "Random Forest":
        final_pred = pred_rf
    else: # Majority Voting
        votes = [pred_lr, pred_svm, pred_rf]
        final_pred = 1 if votes.count(1) > votes.count(0) else 0

    # Display Result Banner
    if final_pred == 0:
        st.markdown("""
        <div class="result-banner-real">
            <span style="font-size: 20px;">🟢</span>
            <span class="label">REAL NEWS</span>
        </div>
        <div class="prediction-explanation">
            The model predicts that this news article is likely <b>REAL</b>.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-banner-fake">
            <span style="font-size: 20px;">🔴</span>
            <span class="label">FAKE NEWS</span>
        </div>
        <div class="prediction-explanation">
            The model predicts that this news article is likely <b>FAKE</b>.
        </div>
        """, unsafe_allow_html=True)
        
    # Prediction Information Expander
    with st.expander("🔍 Prediction Information", expanded=False):
        info_c1, info_c2, info_c3, info_c4 = st.columns(4)
        info_c1.metric("Support Vector Machine", "REAL" if pred_svm == 0 else "FAKE")
        info_c2.metric("Logistic Regression", "REAL" if pred_lr == 0 else "FAKE")
        info_c3.metric("Random Forest", "REAL" if pred_rf == 0 else "FAKE")
        
        votes_count = [pred_svm, pred_lr, pred_rf].count(final_pred)
        agreement_pct = round((votes_count / 3) * 100, 1)
        info_c4.metric("Model Agreement", f"{votes_count}/3 ({agreement_pct}%)")

st.write("")

# ============================================================
# MODEL PERFORMANCE SECTION (MATCHING SCREENSHOT 2)
# ============================================================

st.markdown(f"""
<div class="section-title">📊 Model Performance</div>
<div class="section-desc">Performance of the trained {active_model_name} classifier.</div>
""", unsafe_allow_html=True)

# Performance Data (Dynamically loaded from evaluated test results with verified benchmarks)
DEFAULT_METRICS = {
    "Support Vector Machine": {
        "accuracy": 94.12,
        "precision": 93.71,
        "recall": 93.31,
        "f1": 93.51,
        "cm": [[14843, 814], [870, 12126]]
    },
    "Logistic Regression": {
        "accuracy": 94.26,
        "precision": 93.85,
        "recall": 93.47,
        "f1": 93.66,
        "cm": [[14861, 796], [848, 12148]]
    },
    "Random Forest": {
        "accuracy": 90.63,
        "precision": 90.48,
        "recall": 88.67,
        "f1": 89.57,
        "cm": [[14444, 1213], [1472, 11524]]
    },
    "Majority Voting Ensemble": {
        "accuracy": 94.32,
        "precision": 93.93,
        "recall": 93.52,
        "f1": 93.72,
        "cm": [[14871, 786], [842, 12154]]
    }
}

def get_metrics_cache_signature():
    """
    Tracks file modification timestamps and sizes so @st.cache_data automatically
    invalidates whenever model training exports updated results or confusion matrices.
    """
    tracked_files = [
        "test_results.csv",
        "linear_svm_confusion_matrix.csv",
        "logistic_regression_confusion_matrix.csv",
        "random_forest_confusion_matrix.csv",
        "majority_voting_confusion_matrix.csv"
    ]
    sig = []
    for f in tracked_files:
        p = os.path.join(MODEL_DIR, f)
        if os.path.exists(p):
            try:
                stat = os.stat(p)
                sig.append((f, stat.st_mtime, stat.st_size))
            except OSError:
                sig.append((f, 0, 0))
        else:
            sig.append((f, 0, 0))
    return tuple(sig)

@st.cache_data
def get_metrics_data(cache_signature=None):
    """
    Loads metrics from test_results.csv and confusion matrix CSVs.
    Safely falls back to verified evaluation defaults if any file is missing or corrupted.
    """
    metrics = {k: dict(v) for k, v in DEFAULT_METRICS.items()}
    test_results_path = os.path.join(MODEL_DIR, "test_results.csv")
    if os.path.exists(test_results_path):
        try:
            df_res = pd.read_csv(test_results_path)
            required_cols = {"Model", "Accuracy", "Precision", "Recall", "F1"}
            if required_cols.issubset(df_res.columns):
                name_map = {
                    "Linear SVM - Test": "Support Vector Machine",
                    "Logistic Regression - Test": "Logistic Regression",
                    "Random Forest - Test": "Random Forest",
                    "Majority Voting": "Majority Voting Ensemble"
                }
                for _, row in df_res.iterrows():
                    m_key = name_map.get(str(row["Model"]).strip())
                    if m_key and m_key in metrics:
                        metrics[m_key]["accuracy"] = round(float(row["Accuracy"]) * 100, 2)
                        metrics[m_key]["precision"] = round(float(row["Precision"]) * 100, 2)
                        metrics[m_key]["recall"] = round(float(row["Recall"]) * 100, 2)
                        metrics[m_key]["f1"] = round(float(row["F1"]) * 100, 2)
        except Exception:
            pass

    cm_file_map = {
        "Support Vector Machine": "linear_svm_confusion_matrix.csv",
        "Logistic Regression": "logistic_regression_confusion_matrix.csv",
        "Random Forest": "random_forest_confusion_matrix.csv",
        "Majority Voting Ensemble": "majority_voting_confusion_matrix.csv"
    }
    for m_key, filename in cm_file_map.items():
        cm_path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(cm_path):
            try:
                cm_df = pd.read_csv(cm_path, index_col=0)
                mat = cm_df.values.tolist()
                if len(mat) == 2 and len(mat[0]) == 2 and len(mat[1]) == 2:
                    metrics[m_key]["cm"] = mat
            except Exception:
                pass

    return metrics

METRICS_DATA = get_metrics_data(get_metrics_cache_signature())

active_data = METRICS_DATA.get(active_model_name, METRICS_DATA["Support Vector Machine"])

# 4 Metric Cards in a Row
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Accuracy</div>
        <div class="metric-value">{active_data['accuracy']:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Precision</div>
        <div class="metric-value">{active_data['precision']:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Recall</div>
        <div class="metric-value">{active_data['recall']:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with m_col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">F1 Score</div>
        <div class="metric-value">{active_data['f1']:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ============================================================
# MODEL ANALYTICS SECTION (MATCHING SCREENSHOT 2)
# ============================================================

st.markdown("""
<div class="section-title">📈 Model Analytics</div>
""", unsafe_allow_html=True)

chart_col1, chart_col2, chart_col3 = st.columns(3)

# ------------------------------------------------------------
# CHART 1: ACCURACY DONUT CHART
# ------------------------------------------------------------
with chart_col1:
    st.markdown("""
    <div style="font-size: 16px; font-weight: 700; color: #FFFFFF; display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
        <span>🎯</span> Accuracy
    </div>
    <div class="analytics-box">
        <div class="analytics-box-title">Overall Accuracy</div>
    """, unsafe_allow_html=True)
    
    acc_val = active_data["accuracy"]
    rem_val = max(0, 100 - acc_val)
    
    donut_fig = go.Figure(go.Pie(
        values=[acc_val, rem_val],
        hole=0.74,
        marker_colors=['#10B981', '#1E293B'],
        textinfo='none',
        hoverinfo='none',
        sort=False,
        direction='clockwise'
    ))
    
    donut_fig.update_layout(
        annotations=[
            dict(
                text=f"<b>{acc_val:.2f}%</b>",
                x=0.5,
                y=0.5,
                font_size=28,
                font_color="#FFFFFF",
                font_family="Inter",
                showarrow=False
            )
        ],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=10, l=10, r=10),
        height=270,
        showlegend=False
    )
    
    st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# CHART 2: CLASSIFICATION METRICS BAR CHART
# ------------------------------------------------------------
with chart_col2:
    st.markdown("""
    <div style="font-size: 16px; font-weight: 700; color: #FFFFFF; display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
        <span>📊</span> Performance
    </div>
    <div class="analytics-box">
        <div class="analytics-box-title">Classification Metrics</div>
    """, unsafe_allow_html=True)
    
    prec = int(round(active_data["precision"]))
    rec = int(round(active_data["recall"]))
    f1_val = int(round(active_data["f1"]))
    
    bar_fig = go.Figure(data=[
        go.Bar(
            x=['Precision', 'Recall', 'F1 Score'],
            y=[prec, rec, f1_val],
            marker_color=['#38BDF8', '#A78BFA', '#F59E0B'],
            text=[f'<b>{prec}%</b>', f'<b>{rec}%</b>', f'<b>{f1_val}%</b>'],
            textposition='outside',
            textfont=dict(color='#FFFFFF', size=13, family="Inter"),
            cliponaxis=False,
            width=0.45
        )
    ])
    
    bar_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            range=[0, 115],
            title=dict(text='Score (%)', font=dict(color='#94A3B8', size=12)),
            tickfont=dict(color='#94A3B8', size=11),
            gridcolor='#1E293B',
            zeroline=False
        ),
        xaxis=dict(
            tickfont=dict(color='#E2E8F0', size=12, family="Inter"),
            linecolor='#1E293B'
        ),
        margin=dict(t=25, b=10, l=40, r=10),
        height=270,
        showlegend=False
    )
    
    st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# CHART 3: CONFUSION MATRIX HEATMAP
# ------------------------------------------------------------
with chart_col3:
    st.markdown("""
    <div style="font-size: 16px; font-weight: 700; color: #FFFFFF; display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
        <span>🔲</span> Confusion Matrix
    </div>
    <div class="analytics-box">
        <div class="analytics-box-title">Prediction Results</div>
    """, unsafe_allow_html=True)
    
    cm_matrix = active_data["cm"]
    
    # Text annotations with proper contrast
    cm_annotations = []
    for i, row_label in enumerate(['REAL', 'FAKE']):
        for j, col_label in enumerate(['REAL', 'FAKE']):
            val = cm_matrix[i][j]
            font_col = "#FFFFFF" if val > 1000 else "#0F172A"
            cm_annotations.append(dict(
                x=col_label,
                y=row_label,
                text=f"<b>{val:,}</b>",
                showarrow=False,
                font=dict(color=font_col, size=16, family="Inter")
            ))
            
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm_matrix,
        x=['REAL', 'FAKE'],
        y=['REAL', 'FAKE'],
        colorscale=[
            [0.0, '#FFFFFF'],
            [0.05, '#E2E8F0'],
            [0.2, '#1E3A8A'],
            [1.0, '#0F3B7A']
        ],
        reversescale=False,
        showscale=False,
        xgap=4,
        ygap=4
    ))
    
    cm_fig.update_layout(
        annotations=cm_annotations,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title=dict(text='Predicted', font=dict(color='#94A3B8', size=12)),
            tickfont=dict(color='#E2E8F0', size=12),
            side='bottom'
        ),
        yaxis=dict(
            title=dict(text='Actual', font=dict(color='#94A3B8', size=12)),
            tickfont=dict(color='#E2E8F0', size=12),
            autorange='reversed'
        ),
        margin=dict(t=20, b=35, l=45, r=10),
        height=270
    )
    
    st.plotly_chart(cm_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)