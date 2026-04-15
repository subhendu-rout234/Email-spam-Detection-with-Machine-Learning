"""
app.py — Streamlit Web Application for Email Spam Detection
============================================================
A polished, interactive UI for training ML models on email/SMS data
and predicting whether a given message is Spam or Ham.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import os

from preprocess import preprocess_dataframe, clean_text, get_sample_dataset
from model import train_and_evaluate, predict_text, load_model, model_exists

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Email Spam Detector — ML Powered",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Dark Theme CSS ───────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Import Google Fonts ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ──────────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main container ──────────────────────────────────────────────── */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* ── Hero header ─────────────────────────────────────────────────── */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 18px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .hero-header h1 {
        background: linear-gradient(90deg, #00d2ff 0%, #7b2ff7 50%, #ff6fd8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        color: #b0b3c5;
        font-size: 1.05rem;
        font-weight: 400;
        margin: 0;
    }

    /* ── Metric cards ────────────────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2f, #2a2a40);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(123,47,247,0.15);
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #8a8da0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    /* ── Result badges ───────────────────────────────────────────────── */
    .result-spam {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        padding: 1.2rem 2rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 4px 20px rgba(255,65,108,0.3);
        animation: pulse 2s infinite;
    }
    .result-ham {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 1.2rem 2rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 4px 20px rgba(56,239,125,0.3);
    }

    @keyframes pulse {
        0%   { box-shadow: 0 4px 20px rgba(255,65,108,0.3); }
        50%  { box-shadow: 0 4px 30px rgba(255,65,108,0.55); }
        100% { box-shadow: 0 4px 20px rgba(255,65,108,0.3); }
    }

    /* ── Section headers ─────────────────────────────────────────────── */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e0e0e0;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(123,47,247,0.4);
    }

    /* ── Confidence bar ──────────────────────────────────────────────── */
    .confidence-bar-bg {
        background: rgba(255,255,255,0.08);
        border-radius: 10px;
        height: 14px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.6s ease;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #151527 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ── Info boxes ──────────────────────────────────────────────────── */
    .info-box {
        background: rgba(123,47,247,0.08);
        border: 1px solid rgba(123,47,247,0.25);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: #c4b5fd;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* ── Scrollbar ───────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #1a1a2e; }
    ::-webkit-scrollbar-thumb { background: #7b2ff7; border-radius: 3px; }

    /* ── Divider ─────────────────────────────────────────────────────── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(123,47,247,0.4), transparent);
        margin: 1.5rem 0;
        border: none;
    }

    /* ── Tabs ─────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════
for key in [
    "trained", "results", "best_model", "best_model_name",
    "vectorizer", "processed_df", "models", "all_results",
    "X_test", "y_test",
]:
    if key not in st.session_state:
        st.session_state[key] = None

if "trained" not in st.session_state:
    st.session_state.trained = False


# ═══════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="hero-header">
        <h1>🛡️ Email Spam Detector</h1>
        <p>AI-powered classification using Naive Bayes & Logistic Regression • Built with Scikit-learn & Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — DATA SOURCE
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📂 Data Source")
    data_option = st.radio(
        "Choose dataset",
        ["Upload CSV", "Use Built-in Sample", "Use Default Dataset (Data/spam.csv)"],
        index=2,
        help="Select how you want to provide training data.",
    )

    df_raw = None

    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=["csv"],
            help="CSV must have a text column and a label column (spam/ham).",
        )
        if uploaded_file is not None:
            try:
                df_raw = pd.read_csv(uploaded_file, encoding="latin-1")
                st.success(f"✅ Loaded {len(df_raw):,} rows")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    elif data_option == "Use Built-in Sample":
        df_raw = get_sample_dataset()
        st.info(f"📋 Sample dataset loaded — {len(df_raw)} messages")

    else:
        # Default dataset
        default_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "Data", "spam.csv"
        )
        if os.path.exists(default_path):
            try:
                df_raw = pd.read_csv(default_path, encoding="latin-1")
                st.success(f"✅ Default dataset loaded — {len(df_raw):,} rows")
            except Exception as e:
                st.error(f"Error reading default dataset: {e}")
        else:
            st.warning("Default dataset not found. Please upload a CSV or use the sample.")

    st.markdown("---")

    # ── Training controls ────────────────────────────────────────────────
    st.markdown("### ⚙️ Training Settings")
    test_split = st.slider("Test split ratio", 0.1, 0.4, 0.2, 0.05)

    train_btn = st.button(
        "🚀 Train Models",
        use_container_width=True,
        type="primary",
        disabled=df_raw is None,
    )

    if model_exists():
        st.markdown("---")
        st.markdown(
            '<div class="info-box">💾 A saved model is available on disk and will be '
            "used for predictions if you haven't trained a new one.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; color:#555; font-size:0.75rem; padding-top:0.5rem;">
            Built with ❤️ using Streamlit<br>
            © 2026 Spam Detector ML
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOGIC
# ═══════════════════════════════════════════════════════════════════════════
if train_btn and df_raw is not None:
    with st.spinner("🔧 Preprocessing data & training models…"):
        try:
            result = train_and_evaluate(df_raw, test_size=test_split)

            st.session_state.trained = True
            st.session_state.processed_df = result["processed_df"]
            st.session_state.vectorizer = result["vectorizer"]
            st.session_state.models = result["models"]
            st.session_state.all_results = result["results"]
            st.session_state.best_model_name = result["best_model_name"]
            st.session_state.best_model = result["best_model"]
            st.session_state.X_test = result["X_test"]
            st.session_state.y_test = result["y_test"]

            st.toast("✅ Models trained successfully!", icon="🎉")
        except Exception as e:
            st.error(f"❌ Training failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT — TABS
# ═══════════════════════════════════════════════════════════════════════════
tab_predict, tab_results, tab_data, tab_viz = st.tabs(
    ["🔮 Predict", "📊 Model Results", "📋 Dataset Explorer", "📈 Visualizations"]
)


# ── TAB 1: PREDICT ──────────────────────────────────────────────────────────
with tab_predict:
    st.markdown('<div class="section-header">✉️ Classify a Message</div>', unsafe_allow_html=True)

    col_input, col_result = st.columns([3, 2], gap="large")

    with col_input:
        user_text = st.text_area(
            "Enter an email or SMS message:",
            height=180,
            placeholder="e.g., Congratulations! You've won a $1000 gift card. Call now to claim your prize!",
        )

        predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)

    with col_result:
        if predict_btn:
            if not user_text.strip():
                st.warning("⚠️ Please enter a message to classify.")
            else:
                try:
                    # Use session model if trained, else load from disk
                    model_to_use = st.session_state.best_model
                    vec_to_use = st.session_state.vectorizer

                    if model_to_use is None or vec_to_use is None:
                        if model_exists():
                            model_to_use, vec_to_use = load_model()
                        else:
                            st.error("❌ No model available. Please train a model first.")
                            st.stop()

                    result = predict_text(user_text, model_to_use, vec_to_use)
                    label = result["label"]
                    confidence = result["confidence"]
                    probas = result["probabilities"]

                    # Display result badge
                    if label == "Spam":
                        st.markdown(
                            '<div class="result-spam">🚨 SPAM DETECTED</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="result-ham">✅ HAM (Not Spam)</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Confidence display
                    if confidence > 0:
                        conf_pct = confidence * 100
                        bar_color = (
                            "linear-gradient(90deg, #ff416c, #ff4b2b)"
                            if label == "Spam"
                            else "linear-gradient(90deg, #11998e, #38ef7d)"
                        )
                        st.markdown(
                            f"""
                            <div style="color:#b0b3c5; font-size:0.9rem; margin-bottom:4px;">
                                Confidence: <strong style="color:white">{conf_pct:.1f}%</strong>
                            </div>
                            <div class="confidence-bar-bg">
                                <div class="confidence-bar-fill"
                                     style="width:{conf_pct}%; background:{bar_color};"></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Probability breakdown
                    st.markdown("<br>", unsafe_allow_html=True)
                    prob_col1, prob_col2 = st.columns(2)
                    with prob_col1:
                        st.metric("Ham Probability", f"{probas['Ham']:.2%}")
                    with prob_col2:
                        st.metric("Spam Probability", f"{probas['Spam']:.2%}")

                except FileNotFoundError:
                    st.error("❌ No saved model found. Please train a model first.")
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
        else:
            # Placeholder when no prediction yet
            st.markdown(
                """
                <div style="text-align:center; padding:3rem 1rem; color:#666;">
                    <div style="font-size:3rem; margin-bottom:0.5rem;">🔮</div>
                    <div style="font-size:1rem;">Enter a message and click <strong>Predict</strong><br>
                    to see the classification result here.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Quick test examples
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("**💡 Try these examples:**")
    example_cols = st.columns(3)
    examples = [
        "🎁 Congratulations! You won a FREE iPhone! Click here NOW!",
        "👋 Hey, are we still meeting for lunch tomorrow at noon?",
        "🏆 WINNER!! You've been selected to receive $5000. Call 0800-WIN now!",
    ]
    for i, (col, ex) in enumerate(zip(example_cols, examples)):
        with col:
            st.code(ex, language=None)


# ── TAB 2: MODEL RESULTS ───────────────────────────────────────────────────
with tab_results:
    if not st.session_state.trained:
        st.info("🔧 Train a model first to see results here.")
    else:
        all_results = st.session_state.all_results
        best_name = st.session_state.best_model_name

        st.markdown(
            '<div class="section-header">🏆 Model Comparison</div>',
            unsafe_allow_html=True,
        )

        # ── Metric cards row ─────────────────────────────────────────────
        model_names = list(all_results.keys())
        cols = st.columns(len(model_names))

        for col, name in zip(cols, model_names):
            r = all_results[name]
            is_best = name == best_name
            badge = " 🏅" if is_best else ""
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card" style="{'border: 1px solid rgba(123,47,247,0.5);' if is_best else ''}">
                        <div style="font-size:1rem; font-weight:600; color:#e0e0e0; margin-bottom:1rem;">
                            {name}{badge}
                        </div>
                        <div class="metric-value">{r['accuracy']:.2%}</div>
                        <div class="metric-label">Accuracy</div>
                        <div style="margin-top:1rem; display:flex; justify-content:space-around;">
                            <div>
                                <div style="color:#00d2ff; font-weight:600;">{r['precision']:.2%}</div>
                                <div class="metric-label">Precision</div>
                            </div>
                            <div>
                                <div style="color:#7b2ff7; font-weight:600;">{r['recall']:.2%}</div>
                                <div class="metric-label">Recall</div>
                            </div>
                            <div>
                                <div style="color:#ff6fd8; font-weight:600;">{r['f1']:.2%}</div>
                                <div class="metric-label">F1-Score</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Accuracy comparison chart ────────────────────────────────────
        st.markdown(
            '<div class="section-header">📊 Accuracy Comparison</div>',
            unsafe_allow_html=True,
        )

        comparison_df = pd.DataFrame(
            {
                "Model": model_names,
                "Accuracy": [all_results[n]["accuracy"] for n in model_names],
                "Precision": [all_results[n]["precision"] for n in model_names],
                "Recall": [all_results[n]["recall"] for n in model_names],
                "F1-Score": [all_results[n]["f1"] for n in model_names],
            }
        )

        fig_compare = go.Figure()
        metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
        colors = ["#00d2ff", "#7b2ff7", "#ff6fd8", "#38ef7d"]

        for metric, color in zip(metrics_to_plot, colors):
            fig_compare.add_trace(
                go.Bar(
                    name=metric,
                    x=comparison_df["Model"],
                    y=comparison_df[metric],
                    marker_color=color,
                    text=[f"{v:.2%}" for v in comparison_df[metric]],
                    textposition="outside",
                    textfont=dict(size=12, color="white"),
                )
            )

        fig_compare.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#b0b3c5"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(range=[0, 1.15], gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            height=420,
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # ── Confusion matrices ───────────────────────────────────────────
        st.markdown(
            '<div class="section-header">🔢 Confusion Matrices</div>',
            unsafe_allow_html=True,
        )

        cm_cols = st.columns(len(model_names))
        for col, name in zip(cm_cols, model_names):
            with col:
                cm = all_results[name]["confusion_matrix"]
                fig_cm, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="magma",
                    xticklabels=["Ham", "Spam"],
                    yticklabels=["Ham", "Spam"],
                    ax=ax,
                    cbar_kws={"shrink": 0.8},
                    linewidths=0.5,
                    linecolor="gray",
                    annot_kws={"size": 16, "weight": "bold"},
                )
                ax.set_xlabel("Predicted", fontsize=12, color="white")
                ax.set_ylabel("Actual", fontsize=12, color="white")
                ax.set_title(name, fontsize=13, fontweight="bold", color="white", pad=12)
                ax.tick_params(colors="white")
                fig_cm.patch.set_alpha(0)
                ax.set_facecolor("none")
                st.pyplot(fig_cm)
                plt.close(fig_cm)

        # ── Detailed classification reports ──────────────────────────────
        st.markdown(
            '<div class="section-header">📝 Classification Reports</div>',
            unsafe_allow_html=True,
        )
        for name in model_names:
            with st.expander(f"📄 {name}", expanded=(name == best_name)):
                st.code(all_results[name]["classification_report"], language=None)


# ── TAB 3: DATASET EXPLORER ────────────────────────────────────────────────
with tab_data:
    if st.session_state.processed_df is not None:
        pdf = st.session_state.processed_df

        st.markdown(
            '<div class="section-header">📋 Processed Dataset</div>',
            unsafe_allow_html=True,
        )

        # Stats row
        total = len(pdf)
        n_spam = int(pdf["label"].sum())
        n_ham = total - n_spam

        stat_cols = st.columns(3)
        with stat_cols[0]:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{total:,}</div>
                    <div class="metric-label">Total Messages</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with stat_cols[1]:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value" style="background: linear-gradient(135deg,#38ef7d,#11998e); -webkit-background-clip:text;">{n_ham:,}</div>
                    <div class="metric-label">Ham Messages</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with stat_cols[2]:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value" style="background: linear-gradient(135deg,#ff416c,#ff4b2b); -webkit-background-clip:text;">{n_spam:,}</div>
                    <div class="metric-label">Spam Messages</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Data table
        display_df = pdf[["text", "label", "cleaned_text"]].copy()
        display_df["label"] = display_df["label"].map({0: "Ham ✅", 1: "Spam 🚨"})
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            column_config={
                "text": st.column_config.TextColumn("Original Text", width="large"),
                "label": st.column_config.TextColumn("Label", width="small"),
                "cleaned_text": st.column_config.TextColumn("Cleaned Text", width="large"),
            },
        )

    elif df_raw is not None:
        st.markdown(
            '<div class="section-header">📋 Raw Dataset Preview</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(df_raw.head(50), use_container_width=True, height=400)
        st.info("Train a model to see the fully processed dataset.")
    else:
        st.info("📂 Load a dataset to explore it here.")


# ── TAB 4: VISUALIZATIONS ──────────────────────────────────────────────────
with tab_viz:
    if st.session_state.processed_df is not None:
        pdf = st.session_state.processed_df
        n_spam = int(pdf["label"].sum())
        n_ham = len(pdf) - n_spam

        viz_col1, viz_col2 = st.columns(2, gap="large")

        # ── Label distribution (Donut chart) ─────────────────────────────
        with viz_col1:
            st.markdown(
                '<div class="section-header">📊 Label Distribution</div>',
                unsafe_allow_html=True,
            )
            fig_dist = go.Figure(
                data=[
                    go.Pie(
                        labels=["Ham", "Spam"],
                        values=[n_ham, n_spam],
                        hole=0.55,
                        marker=dict(colors=["#38ef7d", "#ff416c"]),
                        textinfo="label+percent",
                        textfont=dict(size=15, color="white"),
                        hoverinfo="label+value+percent",
                    )
                ]
            )
            fig_dist.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#b0b3c5"),
                showlegend=False,
                height=380,
                margin=dict(t=30, b=30, l=30, r=30),
                annotations=[
                    dict(
                        text=f"<b>{len(pdf):,}</b><br>Total",
                        x=0.5, y=0.5,
                        font_size=18, font_color="white",
                        showarrow=False,
                    )
                ],
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # ── Message length distribution ──────────────────────────────────
        with viz_col2:
            st.markdown(
                '<div class="section-header">📏 Message Length Distribution</div>',
                unsafe_allow_html=True,
            )
            pdf_temp = pdf.copy()
            pdf_temp["msg_length"] = pdf_temp["text"].str.len()
            pdf_temp["label_name"] = pdf_temp["label"].map({0: "Ham", 1: "Spam"})

            fig_len = px.histogram(
                pdf_temp,
                x="msg_length",
                color="label_name",
                color_discrete_map={"Ham": "#38ef7d", "Spam": "#ff416c"},
                nbins=50,
                barmode="overlay",
                opacity=0.7,
                labels={"msg_length": "Message Length (chars)", "label_name": "Label"},
            )
            fig_len.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#b0b3c5"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                height=380,
                margin=dict(t=30, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_len, use_container_width=True)

        # ── Word clouds ──────────────────────────────────────────────────
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">☁️ Word Clouds</div>',
            unsafe_allow_html=True,
        )

        wc_cols = st.columns(2, gap="large")

        for col, (lbl, lbl_name, colormap) in zip(
            wc_cols,
            [(0, "Ham", "Greens"), (1, "Spam", "Reds")],
        ):
            with col:
                st.markdown(f"**{lbl_name} Messages**")
                text_corpus = " ".join(
                    pdf[pdf["label"] == lbl]["cleaned_text"].dropna().tolist()
                )
                if text_corpus.strip():
                    wc = WordCloud(
                        width=600,
                        height=350,
                        background_color=None,
                        mode="RGBA",
                        colormap=colormap,
                        max_words=120,
                        contour_width=0,
                        prefer_horizontal=0.7,
                    ).generate(text_corpus)
                    fig_wc, ax_wc = plt.subplots(figsize=(8, 5))
                    ax_wc.imshow(wc, interpolation="bilinear")
                    ax_wc.axis("off")
                    fig_wc.patch.set_alpha(0)
                    st.pyplot(fig_wc)
                    plt.close(fig_wc)
                else:
                    st.caption("Not enough data to generate a word cloud.")

        # ── Top words bar chart ──────────────────────────────────────────
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">🔤 Top 15 Words by Category</div>',
            unsafe_allow_html=True,
        )

        top_cols = st.columns(2, gap="large")
        for col, (lbl, lbl_name, color) in zip(
            top_cols,
            [(0, "Ham", "#38ef7d"), (1, "Spam", "#ff416c")],
        ):
            with col:
                corpus = " ".join(
                    pdf[pdf["label"] == lbl]["cleaned_text"].dropna().tolist()
                )
                if corpus.strip():
                    from collections import Counter
                    words = corpus.split()
                    top_words = Counter(words).most_common(15)
                    tw_df = pd.DataFrame(top_words, columns=["Word", "Count"])

                    fig_tw = px.bar(
                        tw_df,
                        x="Count",
                        y="Word",
                        orientation="h",
                        color_discrete_sequence=[color],
                        title=f"Top Words — {lbl_name}",
                    )
                    fig_tw.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", color="#b0b3c5"),
                        yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.05)"),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                        height=420,
                        margin=dict(t=40, b=30),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_tw, use_container_width=True)

    else:
        st.info("📊 Train a model to see visualizations here.")


# ═══════════════════════════════════════════════════════════════════════════
# FOOTER (expected CSV format hint)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

with st.expander("📄 Expected CSV Format"):
    st.markdown(
        """
        Your CSV file should have at least two columns:

        | Column Name | Description |
        |-------------|-------------|
        | `v1` / `label` / `class` / `category` | The label column — values should be `spam` or `ham` |
        | `v2` / `text` / `message` / `email` | The text column — the email or SMS content |

        **Example:**
        ```
        v1,v2
        ham,"Hey, are you coming to the party tonight?"
        spam,"FREE entry in a weekly competition! Text WIN to 80080"
        ham,"I'll be there in 5 minutes"
        ```
        The system auto-detects common column names, so most popular SMS/email spam datasets will work out of the box.
        """
    )
