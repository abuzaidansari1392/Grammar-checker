"""
NLP-Based English Grammar Checking and Correction System
=========================================================
Main Streamlit application entry point.

Run with:  streamlit run app.py
"""

import os, sys
# Ensure the project root is on the path so sub-packages import cleanly.
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import time

from nlp.preprocessing  import preprocess_text, tokenize_sentences
from nlp.grammar_detector import detect_all_errors, compute_grammar_score
from nlp.error_classifier import classify_error, get_model_metrics
from nlp.correction       import full_correction_pipeline
from utils.helpers        import (
    summarize_errors, get_error_explanation, get_error_description,
    compute_readability, generate_report,
)
from data.grammar_dataset import ERROR_DESCRIPTIONS

# ------------------------------------------------------------------ #
#  Page configuration
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title  = "AI Grammar Checker",
    page_icon   = "✍️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ------------------------------------------------------------------ #
#  Custom CSS
# ------------------------------------------------------------------ #

st.markdown("""
<style>
  /* ---- Global ---- */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ---- Score gauge card ---- */
  .score-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    color: white;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 20px;
  }
  .score-number {
    font-size: 64px;
    font-weight: 700;
    line-height: 1;
  }
  .score-label {
    font-size: 13px;
    opacity: 0.7;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 6px;
  }

  /* ---- Error cards ---- */
  .error-card {
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 14px;
    border-left: 4px solid;
    background: #f8f9fa;
  }
  .error-type-badge {
    display: inline-block;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
  }
  .error-text-wrong  { color: #c62828; font-weight: 600; }
  .error-text-right  { color: #2e7d32; font-weight: 600; }

  /* ---- Stat chips ---- */
  .stat-chip {
    display: inline-block;
    background: #e3f2fd;
    border-radius: 8px;
    padding: 8px 16px;
    margin: 4px;
    font-size: 14px;
    font-weight: 500;
    color: #1565c0;
  }

  /* ---- Section headers ---- */
  .section-header {
    font-size: 18px;
    font-weight: 600;
    color: #1a1a2e;
    margin: 24px 0 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid #e8eaf6;
  }

  /* ---- Highlighted text area ---- */
  .highlighted-text {
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 18px;
    font-size: 16px;
    line-height: 1.8;
    min-height: 80px;
  }

  /* ---- Tabs ---- */
  .stTabs [data-baseweb="tab"] {
    font-weight: 500;
    font-size: 14px;
  }

  /* ---- Confidence bar ---- */
  .conf-bar-wrap { background: #e0e0e0; border-radius: 4px; height: 8px; }
  .conf-bar-fill { border-radius: 4px; height: 8px; background: #3f51b5; }

  /* ---- Sidebar ---- */
  .sidebar-info {
    background: #e8eaf6;
    border-radius: 10px;
    padding: 14px;
    font-size: 13px;
    margin-top: 12px;
  }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Color map for error types
# ------------------------------------------------------------------ #

ERROR_COLORS = {
    "subject_verb": "#FF6B6B",
    "tense":        "#FFA726",
    "article":      "#AB47BC",
    "preposition":  "#42A5F5",
    "pronoun":      "#26A69A",
    "spelling":     "#EF5350",
    "punctuation":  "#78909C",
    "word_order":   "#8D6E63",
    "correct":      "#66BB6A",
}

SCORE_COLOR = lambda s: "#2e7d32" if s >= 80 else "#f9a825" if s >= 55 else "#c62828"


# ------------------------------------------------------------------ #
#  Cached helpers
# ------------------------------------------------------------------ #

@st.cache_resource(show_spinner="Loading NLP models…")
def load_models():
    """Load all ML/NLP models once and cache them."""
    from nlp.preprocessing   import get_nlp
    from nlp.error_classifier import _load_model
    get_nlp()
    _load_model()
    return True


@st.cache_data(show_spinner=False)
def get_cached_metrics():
    return get_model_metrics()


# ------------------------------------------------------------------ #
#  Core analysis function
# ------------------------------------------------------------------ #

def analyse_text(text: str) -> dict:
    """Run the full analysis pipeline and return all results."""
    # 1. Preprocess
    nlp_data = preprocess_text(text)

    # 2. Detect errors
    errors = detect_all_errors(text, nlp_data["doc"])

    # 3. ML classification per sentence
    sentences    = nlp_data["sentences"]
    ml_results   = [classify_error(s) for s in sentences]

    # 4. Corrections
    correction = full_correction_pipeline(text, errors)

    # 5. Grammar score
    score = compute_grammar_score(text, errors)

    # 6. Error summary
    summary = summarize_errors(errors)

    # 7. Readability
    readability = compute_readability(text)

    return {
        "nlp_data"   : nlp_data,
        "errors"     : errors,
        "ml_results" : ml_results,
        "correction" : correction,
        "score"      : score,
        "summary"    : summary,
        "readability": readability,
    }


# ------------------------------------------------------------------ #
#  Sidebar
# ------------------------------------------------------------------ #

def render_sidebar():
    with st.sidebar:
        st.markdown("## ✍️ AI Grammar Checker")
        st.markdown("---")

        st.markdown("### How it works")
        st.markdown("""
- **spaCy** – tokenization, POS tagging, dependency parsing
- **LanguageTool** – grammar rule matching
- **TF-IDF + LR** – ML error classification
- **pyspellchecker** – spelling verification
        """)

        st.markdown("---")
        st.markdown("### Error Categories")
        for code, name in ERROR_DESCRIPTIONS.items():
            if code == "correct":
                continue
            color = ERROR_COLORS.get(code, "#9e9e9e")
            st.markdown(
                f'<span style="display:inline-block;width:10px;height:10px;'
                f'border-radius:50%;background:{color};margin-right:6px;"></span>{name}',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        if st.button("📊 Show Model Metrics", use_container_width=True):
            st.session_state["show_metrics"] = True

        st.markdown(
            '<div class="sidebar-info">'
            '💡 <strong>Tip:</strong> Enter a sentence or paragraph and click '
            '<em>Check Grammar</em> to analyse it.'
            '</div>',
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------ #
#  Score gauge
# ------------------------------------------------------------------ #

def render_score_gauge(score: float):
    color = SCORE_COLOR(score)
    label = "Excellent" if score >= 90 else "Good" if score >= 75 else \
            "Fair" if score >= 55 else "Needs Work"

    fig = go.Figure(go.Indicator(
        mode   = "gauge+number",
        value  = score,
        number = {"suffix": "/100", "font": {"size": 36, "color": color}},
        gauge  = {
            "axis"      : {"range": [0, 100], "tickwidth": 1},
            "bar"       : {"color": color, "thickness": 0.3},
            "bgcolor"   : "#e0e0e0",
            "borderwidth": 0,
            "steps"     : [
                {"range": [0, 55],  "color": "#ffebee"},
                {"range": [55, 75], "color": "#fff8e1"},
                {"range": [75, 100],"color": "#e8f5e9"},
            ],
            "threshold": {
                "line" : {"color": color, "width": 3},
                "thickness": 0.85,
                "value": score,
            },
        },
        title  = {"text": f"Grammar Score — {label}", "font": {"size": 15}},
    ))
    fig.update_layout(
        height  = 230,
        margin  = dict(t=40, b=0, l=20, r=20),
        paper_bgcolor = "rgba(0,0,0,0)",
        font    = {"family": "Inter"},
    )
    return fig


# ------------------------------------------------------------------ #
#  Error distribution chart
# ------------------------------------------------------------------ #

def render_error_chart(summary: dict):
    by_type = summary["by_type"]
    if not by_type:
        return None

    labels  = [ERROR_DESCRIPTIONS.get(k, k) for k in by_type]
    counts  = list(by_type.values())
    colors  = [ERROR_COLORS.get(k, "#9e9e9e") for k in by_type]

    fig = go.Figure(go.Bar(
        x           = labels,
        y           = counts,
        marker_color= colors,
        text        = counts,
        textposition= "auto",
    ))
    fig.update_layout(
        title          = "Errors by Type",
        xaxis_title    = "Error Category",
        yaxis_title    = "Count",
        height         = 300,
        margin         = dict(t=40, b=60, l=40, r=20),
        paper_bgcolor  = "rgba(0,0,0,0)",
        plot_bgcolor   = "rgba(0,0,0,0)",
        font           = {"family": "Inter", "size": 12},
        xaxis          = {"tickangle": -20},
        yaxis          = {"gridcolor": "#f0f0f0"},
        showlegend     = False,
    )
    return fig


# ------------------------------------------------------------------ #
#  Confusion matrix chart
# ------------------------------------------------------------------ #

def render_confusion_matrix(metrics: dict):
    cm     = metrics["conf_matrix"]
    labels = metrics["label_names"]
    cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100

    fig = px.imshow(
        cm_pct,
        labels      = dict(x="Predicted", y="Actual", color="% of Row"),
        x           = labels,
        y           = labels,
        color_continuous_scale = "Blues",
        text_auto   = ".0f",
        aspect      = "auto",
    )
    fig.update_layout(
        title  = "Confusion Matrix (% of actual class)",
        height = 420,
        font   = {"family": "Inter", "size": 11},
        margin = dict(t=50, b=60, l=100, r=20),
        coloraxis_showscale = False,
    )
    return fig


# ------------------------------------------------------------------ #
#  Individual error card
# ------------------------------------------------------------------ #

def render_error_card(err, idx: int):
    color = ERROR_COLORS.get(err.error_type, "#9e9e9e")
    badge_bg = color + "22"
    desc  = get_error_description(err.error_type)
    expl  = get_error_explanation(err.error_type)
    fix   = err.replacements[0] if err.replacements else "—"
    conf  = int(err.confidence * 100)

    st.markdown(f"""
    <div class="error-card" style="border-left-color:{color};">
        <span class="error-type-badge" style="background:{badge_bg};color:{color};">
            {desc}
        </span>
        <div style="margin:6px 0 4px;">
            <span class="error-text-wrong">✗ "{err.bad_text}"</span>
            &nbsp;→&nbsp;
            <span class="error-text-right">✓ "{fix}"</span>
        </div>
        <div style="font-size:13px;color:#555;margin-bottom:8px;">{err.message}</div>
        <div style="font-size:12px;color:#777;font-style:italic;">💡 {expl}</div>
        <div style="margin-top:10px;">
            <span style="font-size:11px;color:#888;">Confidence: {conf}%</span>
            <div class="conf-bar-wrap" style="margin-top:4px;">
              <div class="conf-bar-fill" style="width:{conf}%;background:{color};"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  NLP details panel
# ------------------------------------------------------------------ #

def render_nlp_details(nlp_data: dict):
    with st.expander("🔬 NLP Analysis Details", expanded=False):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**POS Tags**")
            pos_df = pd.DataFrame(nlp_data["pos_tags"],
                                  columns=["Token", "POS (coarse)", "POS (fine)"])
            st.dataframe(pos_df, use_container_width=True, height=200)

        with c2:
            st.markdown("**Lemmatization**")
            lem_df = pd.DataFrame(nlp_data["lemmas"], columns=["Token", "Lemma"])
            st.dataframe(lem_df, use_container_width=True, height=200)

        st.markdown("**Dependency Parse**")
        dep_df = pd.DataFrame(nlp_data["dependencies"])
        st.dataframe(dep_df, use_container_width=True, height=200)


# ------------------------------------------------------------------ #
#  Main app layout
# ------------------------------------------------------------------ #

def main():
    load_models()
    render_sidebar()

    # ---- Hero header ----
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 24px;">
      <h1 style="font-size:36px;font-weight:700;color:#1a1a2e;margin:0;">
        ✍️ AI Grammar Checker
      </h1>
      <p style="color:#666;font-size:15px;margin-top:6px;">
        NLP-powered grammar analysis · error detection · intelligent correction
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Model metrics modal ----
    if st.session_state.get("show_metrics"):
        with st.container():
            st.markdown("### 📊 ML Model Evaluation")
            with st.spinner("Evaluating model…"):
                metrics = get_cached_metrics()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy",  f"{metrics['accuracy']:.2%}")
            m2.metric("Precision", f"{metrics['precision']:.2%}")
            m3.metric("Recall",    f"{metrics['recall']:.2%}")
            m4.metric("F1 Score",  f"{metrics['f1']:.2%}")

            st.plotly_chart(render_confusion_matrix(metrics),
                            use_container_width=True)

            with st.expander("Classification Report"):
                st.code(metrics["class_report"])

            if st.button("Close metrics"):
                st.session_state["show_metrics"] = False
                st.rerun()
        st.markdown("---")

    # ---- Input area ----
    st.markdown('<p class="section-header">Enter Text</p>', unsafe_allow_html=True)

    sample_texts = {
        "Select a sample…": "",
        "Subject-Verb":    "She don't like coffee and he go to school everyday.",
        "Tense Error":     "Yesterday, I go to the market and buyed some vegetables.",
        "Article Error":   "She is a honest woman and she have a umbrella.",
        "Mixed Errors":    "Him and me went to a university yesterday and we was very tire.",
        "Spelling":        "I definately beleive that the goverment is responsable.",
        "Well-formed":     "She has been working diligently on the project for three months.",
    }
    choice = st.selectbox("Try a sample:", list(sample_texts.keys()))
    default_text = sample_texts[choice]

    user_text = st.text_area(
        "Your text:",
        value       = default_text,
        height      = 160,
        placeholder = "Type or paste English text here…",
        label_visibility = "collapsed",
    )

    col_btn, col_clear, _ = st.columns([1, 1, 6])
    check_btn  = col_btn.button("🔍 Check Grammar", type="primary", use_container_width=True)
    clear_btn  = col_clear.button("🗑️ Clear",        use_container_width=True)

    if clear_btn:
        st.session_state.pop("result", None)
        st.rerun()

    # ---- Analysis ----
    if check_btn:
        if not user_text.strip():
            st.warning("Please enter some text before checking.")
        elif len(user_text.strip()) < 5:
            st.warning("Text is too short for meaningful analysis.")
        else:
            with st.spinner("Analysing grammar…"):
                t0 = time.time()
                result = analyse_text(user_text)
                result["elapsed"] = round(time.time() - t0, 2)
            st.session_state["result"] = result

    # ---- Results ----
    if "result" in st.session_state:
        result   = st.session_state["result"]
        errors   = result["errors"]
        summary  = result["summary"]
        score    = result["score"]
        corr     = result["correction"]
        read     = result["readability"]
        nlp_data = result["nlp_data"]
        ml_res   = result["ml_results"]

        st.markdown("---")

        # Top row: score + stats
        top_left, top_right = st.columns([1, 2])

        with top_left:
            st.plotly_chart(render_score_gauge(score), use_container_width=True)

        with top_right:
            st.markdown("**Text Statistics**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Words",        read["words"])
            c2.metric("Sentences",    read["sentences"])
            c3.metric("Errors Found", summary["total"])
            c4.metric("Readability",  read["readability_level"])

            st.metric("Analysis Time", f"{result['elapsed']}s", label_visibility="visible")

            if summary["total"] == 0:
                st.success("✅ No grammar errors detected! Your text looks great.")
            else:
                st.error(f"⚠️ {summary['total']} grammar issue(s) found.")

        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📝 Corrections",
            "🔴 Error Details",
            "📊 Statistics",
            "🔬 NLP Details",
            "📥 Download",
        ])

        # ---- Tab 1: Corrections ----
        with tab1:
            st.markdown('<p class="section-header">Original Text</p>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="highlighted-text">{corr["highlighted"]}</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<p class="section-header">Corrected Text</p>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="highlighted-text" style="background:#f1f8e9;">'
                f'{corr["corrected"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Sentence-level ML classification
            if ml_res:
                st.markdown('<p class="section-header">Sentence Analysis (ML Classifier)</p>',
                            unsafe_allow_html=True)
                for i, (sent, ml) in enumerate(zip(nlp_data["sentences"], ml_res)):
                    label = ml["label"]
                    conf  = ml["confidence"]
                    color = ERROR_COLORS.get(label, "#9e9e9e")
                    desc  = ERROR_DESCRIPTIONS.get(label, label)
                    st.markdown(
                        f'<div style="padding:10px;border-radius:8px;margin-bottom:8px;'
                        f'border-left:3px solid {color};background:#fafafa;">'
                        f'<span style="font-size:13px;color:#333;">{sent}</span><br>'
                        f'<span style="font-size:11px;color:{color};font-weight:600;">'
                        f'{desc}</span>'
                        f'<span style="font-size:11px;color:#888;margin-left:10px;">'
                        f'confidence: {conf:.0%}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ---- Tab 2: Error Details ----
        with tab2:
            if not errors:
                st.success("✅ No errors found!")
            else:
                st.markdown(f"**{len(errors)} issue(s) detected:**")
                for i, err in enumerate(errors, 1):
                    render_error_card(err, i)

        # ---- Tab 3: Statistics ----
        with tab3:
            if summary["total"] > 0:
                chart = render_error_chart(summary)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)

            # Readability breakdown
            st.markdown('<p class="section-header">Readability</p>',
                        unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            r1.metric("Flesch Score",        read["flesch_score"])
            r2.metric("Avg Words / Sentence", read["avg_words_per_sentence"])
            r3.metric("Syllables",           read["syllables"])

            # Error type breakdown table
            if summary["by_type"]:
                st.markdown('<p class="section-header">Error Breakdown</p>',
                            unsafe_allow_html=True)
                rows = []
                for k, v in summary["by_type"].items():
                    rows.append({
                        "Error Type" : ERROR_DESCRIPTIONS.get(k, k),
                        "Count"      : v,
                        "Explanation": get_error_explanation(k),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # ML probability heatmap (first sentence)
            if ml_res:
                st.markdown('<p class="section-header">ML Classifier — Probability Distribution</p>',
                            unsafe_allow_html=True)
                sent_options = [f"Sentence {i+1}: {s[:50]}…" if len(s) > 50 else f"Sentence {i+1}: {s}"
                                for i, s in enumerate(nlp_data["sentences"])]
                sel = st.selectbox("Select sentence:", sent_options)
                idx = int(sel.split(":")[0].split()[-1]) - 1
                probs = ml_res[idx]["probabilities"]
                fig = go.Figure(go.Bar(
                    x     = [ERROR_DESCRIPTIONS.get(k, k) for k in probs],
                    y     = list(probs.values()),
                    marker_color = [ERROR_COLORS.get(k, "#9e9e9e") for k in probs],
                ))
                fig.update_layout(
                    title        = "Error Type Probabilities",
                    yaxis_title  = "Probability",
                    height       = 300,
                    margin       = dict(t=40, b=60, l=40, r=20),
                    paper_bgcolor= "rgba(0,0,0,0)",
                    plot_bgcolor = "rgba(0,0,0,0)",
                    xaxis        = {"tickangle": -20},
                    yaxis        = {"gridcolor": "#f0f0f0"},
                    font         = {"family": "Inter"},
                )
                st.plotly_chart(fig, use_container_width=True)

        # ---- Tab 4: NLP Details ----
        with tab4:
            render_nlp_details(nlp_data)

        # ---- Tab 5: Download ----
        with tab5:
            st.markdown("### 📥 Export Results")

            report_txt = generate_report(
                original     = user_text,
                corrected    = corr["corrected"],
                errors       = errors,
                grammar_score= score,
                stats        = {**summary, **read},
            )

            col_dl1, col_dl2 = st.columns(2)

            with col_dl1:
                st.download_button(
                    label     = "📄 Download Report (.txt)",
                    data      = report_txt,
                    file_name = "grammar_report.txt",
                    mime      = "text/plain",
                    use_container_width=True,
                )

            with col_dl2:
                st.download_button(
                    label     = "✅ Download Corrected Text",
                    data      = corr["corrected"],
                    file_name = "corrected_text.txt",
                    mime      = "text/plain",
                    use_container_width=True,
                )

            st.markdown("**Preview:**")
            st.code(report_txt, language="text")


# ------------------------------------------------------------------ #
#  Entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    main()
