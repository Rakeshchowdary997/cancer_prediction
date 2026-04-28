import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import joblib
from sklearn.datasets import load_breast_cancer


if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🩺 Cancer Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Breast Cancer Prediction App · Built with Streamlit & scikit-learn"
    }
)


# ════════════════════════════════════════════════════════════════
#  CUSTOM CSS  (theme-aware)
# ════════════════════════════════════════════════════════════════
is_dark = st.session_state.theme == "dark"

# ── Theme tokens ─────────────────────────────────────────────
if is_dark:
    BG_IMAGE     = "https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=1920&q=80"
    OVERLAY      = "rgba(0,0,0,0.60)"
    SIDEBAR_BG   = "rgba(15,20,30,0.88)"
    MAIN_BG      = "rgba(10,15,25,0.65)"
    TEXT_COLOR   = "#f0f0f0"
    SUBTEXT      = "#adb5bd"
    CARD_BG      = "rgba(30,35,50,0.80)"
    CARD_BORDER  = "#2e3a4a"
    INPUT_BG     = "rgba(20,28,40,0.90)"
    DIVIDER      = "#2e3a4a"
    BTN_BG       = "rgba(255,255,255,0.10)"
    BTN_COLOR    = "#f0f0f0"
    BTN_BORDER   = "rgba(255,255,255,0.25)"
# REPLACE the else block with this:
else:
    BG_IMAGE     = "https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=1920&q=80"
    OVERLAY      = "rgba(180, 210, 240, 0.55)"   # less white, more tinted
    SIDEBAR_BG   = "rgba(225, 235, 250, 0.97)"
    MAIN_BG      = "rgba(240, 245, 255, 0.90)"
    TEXT_COLOR   = "#0d1b2a"                      # deep navy — always visible
    SUBTEXT      = "#2c3e50"                      # darker subtext
    CARD_BG      = "rgba(255,255,255,0.95)"
    CARD_BORDER  = "#a8c0e0"
    INPUT_BG     = "rgba(255,255,255,0.98)"
    DIVIDER      = "#a8c0e0"
    BTN_BG       = "rgba(0,0,0,0.08)"
    BTN_COLOR    = "#0d1b2a"
    BTN_BORDER   = "rgba(0,0,0,0.25)"

st.markdown(f"""
<style>
/* ── Background ── */
[data-testid="stAppViewContainer"] {{
    background-image: url("{BG_IMAGE}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: {OVERLAY};
    z-index: 0;
}}
[data-testid="stAppViewContainer"] > * {{
    position: relative;
    z-index: 1;
}}

/* ── Sidebar ── */
/* ── Expander headers — full override ── */
[data-testid="stExpander"] details summary,
[data-testid="stExpander"] details summary *,
[data-testid="stExpander"] details summary p,
[data-testid="stExpander"] details summary span,
[data-testid="stExpander"] details > summary > div > div > p,
div[data-testid="stExpander"] summary div p,
div[data-testid="stExpander"] summary span,
button[data-testid="stExpanderToggleIcon"],
[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
[data-testid="stSidebar"] [data-testid="stExpander"] summary span,
[data-testid="stSidebar"] details summary *  {{
    color: {TEXT_COLOR} !important;
    font-weight: 600 !important;
    opacity: 1 !important;
    visibility: visible !important;
}}

/* ── Expander arrow ── */
[data-testid="stExpander"] svg,
[data-testid="stSidebar"] [data-testid="stExpander"] svg {{
    fill: {TEXT_COLOR} !important;
    stroke: {TEXT_COLOR} !important;
    opacity: 1 !important;
}}

/* ── Expander background ── */
[data-testid="stExpander"],
[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background: {CARD_BG} !important;
    border: 1px solid {CARD_BORDER} !important;
    border-radius: 10px !important;
}}
[data-testid="stSidebar"] {{
    background: {SIDEBAR_BG} !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid {CARD_BORDER};
}}
[data-testid="stSidebar"] * {{
    color: {TEXT_COLOR} !important;
}}

/* ── Main block ── */
[data-testid="stMainBlockContainer"] {{
    background: {MAIN_BG} !important;
    backdrop-filter: blur(6px);
    border-radius: 16px;
    padding: 1.5rem !important;
}}

html, body, [data-testid="stAppViewContainer"],
.stMarkdown, .stMarkdown p, .stMarkdown h1,
.stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
.stMarkdown h5, .stMarkdown h6,
label, caption, small,
[data-testid="stMetricLabel"], [data-testid="stMetricValue"],
[data-testid="stCaptionContainer"] p,
[data-testid="stMarkdownContainer"] p {{
    color: {TEXT_COLOR} !important;
}}
/* ── Tabs ── */
[data-testid="stTabs"] button {{
    color: {TEXT_COLOR} !important;
    background: transparent !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    border-bottom: 3px solid #4a90d9 !important;
    color: #4a90d9 !important;
}}

/* ── Inputs & sliders ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
.stSelectbox div[data-baseweb="select"] {{
    background: {INPUT_BG} !important;
    color: {TEXT_COLOR} !important;
    border: 1px solid {CARD_BORDER} !important;
    border-radius: 8px !important;
}}

/* ── Selectbox text fix ── */
[data-testid="stSelectbox"] div[data-baseweb="select"] *,
[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] div {{
    color: {TEXT_COLOR} !important;
    background-color: transparent !important;
}}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    background-color: {INPUT_BG} !important;
}}

/* ── Dropdown popup options ── */
[data-baseweb="popover"] *,
[data-baseweb="menu"] *,
ul[data-baseweb="menu"] li {{
    color: {TEXT_COLOR} !important;
    background-color: {INPUT_BG} !important;
}}
ul[data-baseweb="menu"] li:hover {{
    background-color: {CARD_BG} !important;
    filter: brightness(0.95);
}}

/* ── Dropdown arrow ── */
[data-testid="stSelectbox"] svg {{
    fill: {TEXT_COLOR} !important;
}}

/* ── Buttons ── */
button[kind="secondary"], button[data-testid="baseButton-secondary"] {{
    background: {BTN_BG} !important;
    color: {BTN_COLOR} !important;
    border: 1px solid {BTN_BORDER} !important;
    border-radius: 20px !important;
    font-size: 0.82rem !important;
    padding: 4px 14px !important;
}}

/* ── Cards ── */
.model-card {{
    background: {CARD_BG} !important;
    border: 1px solid {CARD_BORDER} !important;
    color: {TEXT_COLOR} !important;
    border-radius: 12px;
    padding: 1rem 0.8rem;
    text-align: center;
}}
.about-box {{
    background: {CARD_BG} !important;
    border: 1px solid {CARD_BORDER} !important;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 10px;
    color: {TEXT_COLOR} !important;
}}
.about-box p {{ color: {SUBTEXT} !important; }}

/* ── XAI cards ── */
.xai-warning {{
    background: #fff3cd; color: #856404;
    padding: 0.6rem 1rem; border-radius: 10px;
    border-left: 5px solid #ffc107;
    font-size: 0.88rem; margin-bottom: 6px;
}}
.xai-safe {{
    background: #d1ecf1; color: #0c5460;
    padding: 0.6rem 1rem; border-radius: 10px;
    border-left: 5px solid #17a2b8;
    font-size: 0.88rem; margin-bottom: 6px;
}}

/* ── Result banners ── */
.result-benign {{
    background: #d4edda; color: #155724;
    padding: 1.2rem 1.5rem; border-radius: 14px;
    border-left: 6px solid #28a745;
    font-size: 1.25rem; font-weight: 700; margin-bottom: 1rem;
}}
.result-malignant {{
    background: #f8d7da; color: #721c24;
    padding: 1.2rem 1.5rem; border-radius: 14px;
    border-left: 6px solid #dc3545;
    font-size: 1.25rem; font-weight: 700; margin-bottom: 1rem;
}}

/* ── Divider ── */
hr {{ border-color: {DIVIDER} !important; }}

/* ── Table ── */
table {{ color: {TEXT_COLOR} !important; }}
th, td {{ border-color: {CARD_BORDER} !important; color: {TEXT_COLOR} !important; }}

/* ── Sidebar number input labels ── */
div[data-testid="stNumberInput"] label {{ font-size: 0.75rem !important; }}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  LOAD ASSETS
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def load_assets():
    lr  = joblib.load("model_lr.pkl")
    rf  = joblib.load("model_rf.pkl")
    svm = joblib.load("model_svm.pkl")
    sc  = joblib.load("scaler.pkl")
    res = joblib.load("model_results.pkl")
    fn  = joblib.load("feature_names.pkl")
    return lr, rf, svm, sc, res, fn

try:
    lr_model, rf_model, svm_model, scaler, model_results, feature_names = load_assets()
    models_loaded = True
except FileNotFoundError:
    models_loaded = False

cancer_data = load_breast_cancer()
df_raw      = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)

MODEL_MAP = {}
if models_loaded:
    MODEL_MAP = {
        "Logistic Regression": lr_model,
        "Random Forest":       rf_model,
        "SVM":                 svm_model,
    }

# ════════════════════════════════════════════════════════════════
#  HERO HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("# 🩺 Breast Cancer Prediction")
st.markdown(
    "An **Explainable AI** tool that predicts whether a breast tumour is "
    "**Benign** ✅ or **Malignant** ⚠️ using three ML models."
)
st.divider()

# ── Theme toggle ─────────────────────────────────────────────
col_space, col_toggle = st.columns([6, 1])
with col_toggle:
    if st.session_state.theme == "dark":
        if st.button("☀️ Light", key="theme_btn"):
            st.session_state.theme = "light"
            st.rerun()
    else:
        if st.button("🌙 Dark", key="theme_btn"):
            st.session_state.theme = "dark"
            st.rerun()

if not models_loaded:
    st.error(
        "⚠️ Model `.pkl` files not found.  \n"
        "Run `python train_model.py` in your terminal first, then restart the app."
    )
    st.stop()

# ════════════════════════════════════════════════════════════════
#  SIDEBAR — CONFIGURATION + INPUT FORM
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    selected_model_name = st.selectbox(
        "🤖 Choose Model",
        list(MODEL_MAP.keys()),
        help="Select the ML model used for prediction"
    )
    active_model = MODEL_MAP[selected_model_name]

    st.divider()
    st.markdown("## 📋 Cell Nucleus Measurements")
    st.caption("Use sliders or type exact values in the boxes.")

    mean_feats  = [f for f in feature_names if "mean"  in f]
    se_feats    = [f for f in feature_names if "error" in f]
    worst_feats = [f for f in feature_names if "worst" in f]

    def render_inputs(feat_list):
        vals = {}
        for f in feat_list:
            fmin  = float(df_raw[f].min())
            fmax  = float(df_raw[f].max())
            fmean = float(df_raw[f].mean())
            c1, c2 = st.columns([3, 1])
            with c1:
                sv = st.slider(f, fmin, fmax, fmean, key=f"sl_{f}")
            with c2:
                nv = st.number_input(
                    "val", min_value=fmin, max_value=fmax,
                    value=sv, key=f"ni_{f}", label_visibility="collapsed"
                )
            vals[f] = nv
        return vals

    with st.expander("📊 Mean Features", expanded=True):
        mean_vals = render_inputs(mean_feats)
    with st.expander("📉 Standard Error", expanded=False):
        se_vals = render_inputs(se_feats)
    with st.expander("⚠️ Worst (Largest) Values", expanded=False):
        worst_vals = render_inputs(worst_feats)

    st.divider()
    predict_btn = st.button("🔍 Run Prediction", use_container_width=True, type="primary")

# ── Assemble full input vector ────────────────────────────────
all_vals    = {**mean_vals, **se_vals, **worst_vals}
input_array = np.array([all_vals[f] for f in feature_names]).reshape(1, -1)
scaled_inp  = scaler.transform(input_array)

# Convenience shorthands used by XAI
v = all_vals  # alias

# ════════════════════════════════════════════════════════════════
#  MAIN TABS
# ════════════════════════════════════════════════════════════════
tab_pred, tab_xai, tab_fi, tab_cmp, tab_about = st.tabs([
    "🔬 Prediction",
    "🧠 Explainability (XAI)",
    "📊 Feature Importance",
    "🏆 Model Comparison",
    "📜 About Project",
])

# ════════════════════════════════════════════════════════════════
#  TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════════════
with tab_pred:
    if predict_btn:
        pred  = active_model.predict(scaled_inp)[0]
        proba = active_model.predict_proba(scaled_inp)[0]

        # ── Result banner ─────────────────────────────────
        if pred == 1:
            st.markdown(
                "<div class='result-benign'>✅ Benign — No Cancer Risk Detected (Safe)</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-malignant'>⚠️ Malignant — Cancer Risk Detected</div>",
                unsafe_allow_html=True
            )

        col_info, col_chart = st.columns([1, 1])

        with col_info:
            st.markdown("##### 📌 Prediction Details")
            st.markdown(f"- **Model:** {selected_model_name}")
            st.markdown(f"- **Accuracy (test set):** {model_results[selected_model_name]*100:.2f}%")
            st.markdown(f"- **Malignant probability:** {proba[0]*100:.1f}%")
            st.markdown(f"- **Benign probability:** {proba[1]*100:.1f}%")

        with col_chart:
            fig, ax = plt.subplots(figsize=(4.5, 2.5))
            labels = ["Malignant", "Benign"]
            vals_p = [proba[0] * 100, proba[1] * 100]
            colors = ["#e74c3c", "#2ecc71"]
            bars   = ax.barh(labels, vals_p, color=colors, height=0.45)
            for b, p in zip(bars, vals_p):
                ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2,
                        f"{p:.1f}%", va="center", fontsize=11, fontweight="bold")
            ax.set_xlim(0, 115)
            ax.set_xlabel("Confidence (%)", fontsize=10)
            ax.spines[["top", "right", "left"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # ── All 3 models quick glance ─────────────────────
        st.divider()
        st.markdown("##### 🤖 All-Model Snapshot")
        c1, c2, c3 = st.columns(3)
        for col, (mname, mdl) in zip([c1, c2, c3], MODEL_MAP.items()):
            mp   = mdl.predict(scaled_inp)[0]
            mprb = mdl.predict_proba(scaled_inp)[0]
            lbl  = "✅ Benign" if mp == 1 else "⚠️ Malignant"
            clr  = "#2ecc71"  if mp == 1 else "#e74c3c"
            conf = mprb[mp] * 100
            col.markdown(f"""
            <div class='model-card'>
                <h4>{mname}</h4>
                <div class='pred' style='color:{clr}'>{lbl}</div>
                <div class='conf'>{conf:.1f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info(
            "👈 Adjust the **cell nucleus measurements** in the sidebar "
            "and click **Run Prediction** to see results."
        )

# ════════════════════════════════════════════════════════════════
#  TAB 2 — EXPLAINABILITY (XAI)
# ════════════════════════════════════════════════════════════════
with tab_xai:
    st.markdown("### 🧠 Real-Time Prediction Explanation")
    st.caption(
        "These rules interpret each feature value and explain "
        "why certain measurements increase or decrease cancer risk."
    )

    # ── XAI rules ────────────────────────────────────────────
    xai_findings = []

    # Mean radius
    if v["mean radius"] > 17:
        xai_findings.append(("warn", "🔴 **Mean Radius** is very high (> 17 µm) — strongly associated with malignancy."))
    elif v["mean radius"] > 13:
        xai_findings.append(("warn", "🟠 **Mean Radius** is elevated (> 13 µm) — increases cancer risk."))
    else:
        xai_findings.append(("safe", "🟢 **Mean Radius** is within a normal range — low risk indicator."))

    # Mean texture
    if v["mean texture"] > 21:
        xai_findings.append(("warn", "🔴 **Mean Texture** is high (> 21) — irregular cell surfaces are a risk factor."))
    else:
        xai_findings.append(("safe", "🟢 **Mean Texture** is normal — smooth cell surfaces are a positive sign."))

    # Mean perimeter
    if v["mean perimeter"] > 110:
        xai_findings.append(("warn", "🔴 **Mean Perimeter** is high (> 110 µm) — large perimeter may indicate malignant growth."))
    elif v["mean perimeter"] > 85:
        xai_findings.append(("warn", "🟠 **Mean Perimeter** is slightly elevated (> 85 µm)."))
    else:
        xai_findings.append(("safe", "🟢 **Mean Perimeter** is within a safe range."))

    # Mean area
    if v["mean area"] > 800:
        xai_findings.append(("warn", "🔴 **Mean Area** is large (> 800 µm²) — larger nuclei are associated with malignancy."))
    else:
        xai_findings.append(("safe", "🟢 **Mean Area** is normal (≤ 800 µm²)."))

    # Mean smoothness
    if v["mean smoothness"] > 0.12:
        xai_findings.append(("warn", "🟠 **Mean Smoothness** is high (> 0.12) — irregular cell boundaries detected."))
    else:
        xai_findings.append(("safe", "🟢 **Mean Smoothness** is in the healthy range."))

    # Mean compactness
    if v["mean compactness"] > 0.15:
        xai_findings.append(("warn", "🟠 **Mean Compactness** is elevated (> 0.15) — compact, irregular nuclei are a concern."))
    else:
        xai_findings.append(("safe", "🟢 **Mean Compactness** is normal."))

    # Mean concavity
    if v["mean concavity"] > 0.15:
        xai_findings.append(("warn", "🔴 **Mean Concavity** is high (> 0.15) — deep concavities are a strong malignancy indicator."))
    elif v["mean concavity"] > 0.08:
        xai_findings.append(("warn", "🟠 **Mean Concavity** is slightly elevated."))
    else:
        xai_findings.append(("safe", "🟢 **Mean Concavity** is within a normal range."))

    # Mean concave points
    if v["mean concave points"] > 0.1:
        xai_findings.append(("warn", "🔴 **Mean Concave Points** is high (> 0.1) — multiple concave segments in the nucleus."))
    else:
        xai_findings.append(("safe", "🟢 **Mean Concave Points** appears normal."))

    # Worst radius
    if v["worst radius"] > 20:
        xai_findings.append(("warn", "🔴 **Worst Radius** is very large (> 20 µm) — the largest nuclei measured are significantly enlarged."))
    else:
        xai_findings.append(("safe", "🟢 **Worst Radius** is within an acceptable range."))

    # Worst concave points
    if v["worst concave points"] > 0.18:
        xai_findings.append(("warn", "🔴 **Worst Concave Points** is very high (> 0.18) — critical risk factor."))
    elif v["worst concave points"] > 0.12:
        xai_findings.append(("warn", "🟠 **Worst Concave Points** is elevated."))
    else:
        xai_findings.append(("safe", "🟢 **Worst Concave Points** is in a safe range."))

    # ── Summary counts ────────────────────────────────────
    n_warn = sum(1 for kind, _ in xai_findings if kind == "warn")
    n_safe = sum(1 for kind, _ in xai_findings if kind == "safe")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("🔴 Risk Signals",   n_warn)
    col_b.metric("🟢 Safe Signals",   n_safe)
    col_c.metric("📋 Features Checked", len(xai_findings))

    st.markdown("---")
    st.markdown("#### 🔍 Feature-by-Feature Analysis")

    for kind, msg in xai_findings:
        css_class = "xai-warning" if kind == "warn" else "xai-safe"
        st.markdown(f"<div class='{css_class}'>{msg}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "> ⚠️ **Medical Disclaimer:** This tool is for educational purposes only. "
        "It is **not** a substitute for professional medical diagnosis."
    )

# ════════════════════════════════════════════════════════════════
#  TAB 3 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════
with tab_fi:
    st.markdown("### 📊 Feature Importance Visualization")

    col_sel, col_n = st.columns([1.5, 1])
    with col_sel:
        fi_choice = st.selectbox(
            "Model for importance",
            ["Random Forest", "Logistic Regression"],
            key="fi_sel"
        )
    with col_n:
        top_n = st.slider("Top N features", 5, 30, 15, key="top_n")

    if fi_choice == "Random Forest":
        importances = rf_model.feature_importances_
        note = "Mean Decrease in Impurity (MDI)"
        note_extra = "Random Forest naturally ranks features by how much each one reduces impurity across all trees."
    else:
        importances = np.abs(lr_model.coef_[0])
        note = "Absolute Logistic Regression Coefficient"
        note_extra = "Larger absolute coefficients mean the feature has a stronger influence on the log-odds of prediction."

    st.caption(f"📌 **Metric:** {note} — {note_extra}")

    indices  = np.argsort(importances)[::-1][:top_n]
    top_feat = [feature_names[i] for i in indices]
    top_imp  = importances[indices]

    def bar_color(name):
        if "mean"  in name: return "#4a90d9"
        if "error" in name: return "#f39c12"
        return "#e74c3c"

    colors = [bar_color(f) for f in top_feat]

    fig, ax = plt.subplots(figsize=(9, top_n * 0.42 + 1.2))
    y_pos = range(top_n)
    bars  = ax.barh(
        list(y_pos),
        top_imp[::-1],
        color=colors[::-1],
        height=0.62,
        edgecolor="white"
    )
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f[:35] for f in top_feat[::-1]], fontsize=9)
    ax.set_xlabel(note, fontsize=10)
    ax.set_title(f"Top {top_n} Features — {fi_choice}", fontsize=12, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)

    # Value labels
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(top_imp) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", fontsize=8, color="#333")

      # Legend
    patches = [
        mpatches.Patch(color="#4a90d9", label="Mean features"),
        mpatches.Patch(color="#f39c12", label="Std error features"),
        mpatches.Patch(color="#e74c3c", label="Worst features"),
    ]
    ax.legend(handles=patches, fontsize=9, loc="lower right")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Top 5 callout ─────────────────────────────────────
    st.markdown("#### 🏅 Top 5 Most Important Features")
    for i, (feat, imp) in enumerate(zip(top_feat[:5], top_imp[:5])):
        rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
        bar_len    = int(imp / top_imp[0] * 20)
        bar_vis    = "█" * bar_len + "░" * (20 - bar_len)
        st.markdown(f"{rank_emoji} **{feat}** — `{bar_vis}` `{imp:.5f}`")

# ════════════════════════════════════════════════════════════════
#  TAB 4 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════
with tab_cmp:
    st.markdown("### 🏆 Model Comparison")
    st.caption("Accuracy evaluated on 30% held-out test data (random_state=2529).")

    # ── Accuracy bar chart ────────────────────────────────
    names  = list(model_results.keys())
    accs   = [model_results[n] * 100 for n in names]
    colors_cmp = ["#4a90d9", "#2ecc71", "#e67e22"]

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(names, accs, color=colors_cmp, height=0.45)
    for b, a in zip(bars, accs):
        ax.text(b.get_width() + 0.2, b.get_y() + b.get_height() / 2,
                f"{a:.2f}%", va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(85, 102)
    ax.set_xlabel("Test Accuracy (%)", fontsize=10)
    ax.set_title("Model Test Accuracy", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Metric cards ──────────────────────────────────────
    st.markdown("#### 📋 Quick Summary")
    c1, c2, c3 = st.columns(3)
    icons = ["🔵", "🟢", "🟠"]
    descs = [
        "Linear model, fast and interpretable.",
        "Ensemble of trees, handles non-linearity.",
        "Margin-based, robust on small datasets."
    ]
    for col, (name, acc), icon, desc in zip([c1, c2, c3], zip(names, accs), icons, descs):
        col.markdown(f"""
        <div class='about-box' style='text-align:center'>
            <h4>{icon} {name}</h4>
            <p style='font-size:1.6rem;font-weight:700;margin:4px 0'>{acc:.2f}%</p>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Pros & Cons table ─────────────────────────────────
    st.markdown("#### ⚖️ Pros & Cons")
    comparison_data = {
        "Model":         ["Logistic Regression", "Random Forest", "SVM"],
        "Speed":         ["⚡ Very Fast",         "🐢 Moderate",   "🐢 Moderate"],
        "Interpretable": ["✅ Yes",               "⚠️ Partial",    "❌ No"],
        "Feature Imp.":  ["✅ Coefficients",      "✅ MDI",         "❌ Not direct"],
        "Best For":      ["Baseline / simple",   "Complex data",  "Small / clean data"],
    }
    df_cmp = pd.DataFrame(comparison_data).set_index("Model")
    st.table(df_cmp)

# ════════════════════════════════════════════════════════════════
#  TAB 5 — ABOUT PROJECT
# ════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### 📜 About This Project")

    # ── Dataset ───────────────────────────────────────────
    st.markdown("""
    <div class='about-box'>
        <h4>📦 Dataset — Wisconsin Breast Cancer Dataset</h4>
        <p>
        Originally from the <strong>UCI Machine Learning Repository</strong>,
        this dataset contains features computed from a digitized image of a
        <em>Fine Needle Aspirate (FNA)</em> of a breast mass.
        <br><br>
        • <strong>569 samples</strong> — 357 Benign, 212 Malignant<br>
        • <strong>30 numeric features</strong> derived from cell nucleus measurements<br>
        • <strong>Target:</strong> Diagnosis — M (Malignant) or B (Benign)<br>
        • <strong>Features:</strong> radius, texture, perimeter, area, smoothness,
          compactness, concavity, concave points, symmetry, fractal dimension
          (× 3 statistics = mean, std error, worst)<br>
        • <strong>No missing values</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Algorithms ────────────────────────────────────────
    st.markdown("""
    <div class='about-box'>
        <h4>🤖 Algorithms Used</h4>
        <p>
        <strong>1. Logistic Regression</strong> — A linear classification model that estimates the
        probability of a binary outcome using the logistic (sigmoid) function.
        Ideal as a fast, interpretable baseline.<br><br>
        <strong>2. Random Forest</strong> — An ensemble of 100 decision trees, each trained on a
        random subset of data and features. Reduces variance and handles non-linear
        decision boundaries well. Provides native feature importance via Mean Decrease in Impurity.<br><br>
        <strong>3. Support Vector Machine (SVM)</strong> — Finds the optimal hyperplane that
        maximises the margin between classes. Uses an RBF kernel for non-linear data.
        Robust on small, clean datasets.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Accuracy ──────────────────────────────────────────
    st.markdown("#### 🎯 Model Accuracy (70/30 Train–Test Split)")
    for name, acc in model_results.items():
        bar_len = int(acc * 30)
        bar_vis = "█" * bar_len + "░" * (30 - bar_len)
        st.markdown(f"**{name}**  `{bar_vis}`  **{acc*100:.2f}%**")

    # ── Pipeline ──────────────────────────────────────────
    st.markdown("""
    <div class='about-box'>
        <h4>🔁 ML Pipeline</h4>
        <p>
        1. Load Wisconsin dataset from <code>sklearn.datasets</code><br>
        2. 70/30 stratified train-test split (<code>random_state=2529</code>)<br>
        3. Feature scaling with <code>StandardScaler</code><br>
        4. Train 3 models and evaluate on held-out test set<br>
        5. Save models and scaler with <code>joblib</code><br>
        6. Streamlit app loads saved artifacts for real-time inference<br>
        7. XAI layer applies domain-informed rules on raw feature values
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Tech stack ────────────────────────────────────────
    st.markdown("""
    <div class='about-box'>
        <h4>🛠️ Tech Stack</h4>
        <p>
        Python · Streamlit · scikit-learn · NumPy · Pandas · Matplotlib · Joblib
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "> ⚠️ **Disclaimer:** This application is built for **educational purposes** only. "
        "It is **not a medical device** and must not be used for clinical diagnosis. "
        "Always consult a qualified medical professional."
    )
    st.caption("Built with ❤️ using Streamlit · YBI Foundation Project")


import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))