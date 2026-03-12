"""
MILESTONE 7: Streamlit Frontend — Exam Anxiety Detector
AI-Based Exam Anxiety Detector
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
from datetime import datetime

from model.predictor import get_predictor, ANXIETY_COLORS, ANXIETY_EMOJIS, ANXIETY_TIPS

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Exam Anxiety Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1117 100%);
}

.main-header {
    text-align: center;
    padding: 2rem 0 1rem;
}

.main-header h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #2dd4bf, #818cf8, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}

.main-header p {
    color: #6b7280;
    font-size: 1rem;
    font-weight: 300;
}

.result-card {
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(12px);
}

.result-emoji {
    font-size: 5rem;
    display: block;
    margin-bottom: 0.5rem;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.07); }
}

.result-label {
    font-size: 2rem;
    font-weight: 700;
    margin: 0.5rem 0;
}

.confidence-badge {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    color: #9ca3af;
    margin-top: 0.5rem;
}

.tips-container {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1.5rem;
}

.tip-item {
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: #d1d5db;
    font-size: 0.95rem;
    line-height: 1.5;
}

.tip-item:last-child { border-bottom: none; }

.disclaimer-box {
    background: rgba(249, 115, 22, 0.08);
    border: 1px solid rgba(249, 115, 22, 0.3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 1.5rem;
    color: #fdba74;
    font-size: 0.85rem;
    line-height: 1.5;
}

.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}

.sidebar-info {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 1rem;
    font-size: 0.85rem;
    color: #9ca3af;
    line-height: 1.6;
}

.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #f3f4f6 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.95rem !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2dd4bf, #818cf8) !important;
    color: #0d1117 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-size: 1rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(45, 212, 191, 0.3) !important;
}

.history-item {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    border-left: 3px solid;
    font-size: 0.85rem;
}

div[data-testid="stSidebar"] {
    background: rgba(13,17,23,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor():
    return get_predictor()


def render_gauge(confidence: float, label: str) -> go.Figure:
    color = ANXIETY_COLORS[label]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        number={"suffix": "%", "font": {"color": "white", "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4b5563", "tickfont": {"color": "#9ca3af"}},
            "bar": {"color": color},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33], "color": "rgba(45,212,191,0.08)"},
                {"range": [33, 66], "color": "rgba(245,158,11,0.08)"},
                {"range": [66, 100], "color": "rgba(248,113,113,0.08)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": confidence * 100,
            },
        },
        title={"text": "Confidence Score", "font": {"color": "#9ca3af", "size": 14}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(t=30, b=0, l=20, r=20),
    )
    return fig


def render_probability_bar(probabilities: dict) -> go.Figure:
    labels = list(probabilities.keys())
    values = [round(v * 100, 1) for v in probabilities.values()]
    colors = [ANXIETY_COLORS[l] for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont=dict(color="white", size=12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=180,
        margin=dict(t=10, b=10, l=10, r=60),
        xaxis=dict(
            range=[0, 110], showgrid=False,
            zeroline=False, showticklabels=False,
            color="#9ca3af",
        ),
        yaxis=dict(color="white", tickfont=dict(size=12)),
        showlegend=False,
    )
    return fig


def render_distribution_pie(distribution: dict) -> go.Figure:
    labels = list(distribution.keys())
    values = list(distribution.values())
    colors = [ANXIETY_COLORS[l] for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors, line=dict(color="#0d1117", width=2)),
        textinfo="label+percent",
        textfont=dict(color="white", size=11),
        hole=0.45,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Exam Anxiety Detector")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🔍 Single Analysis", "📊 Batch Analysis", "📈 History & Trends"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    predictor = load_predictor()
    model_status = "🟢 BERT Model" if predictor._loaded else "🟡 Rule-Based"
    st.markdown(f"**Model:** {model_status}")
    st.markdown(f"**Device:** {'GPU' if predictor.device.type == 'cuda' else 'CPU'}")

    st.markdown("---")
    st.markdown("""
    <div class="sidebar-info">
    <strong>About this tool</strong><br><br>
    This AI-powered system identifies exam-related anxiety from student text using NLP and BERT.<br><br>
    <strong>Anxiety Levels:</strong><br>
    🟢 Low Anxiety<br>
    🟡 Moderate Anxiety<br>
    🔴 High Anxiety<br><br>
    <strong>⚠️ Non-Diagnostic:</strong> This tool is for supportive purposes only and does not replace professional mental health assessment.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SINGLE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
if "Single" in page:
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Exam Anxiety Detector</h1>
        <p>AI-powered emotional wellness screening for students</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("#### ✍️ Share Your Thoughts")
        user_text = st.text_area(
            "How are you feeling about your upcoming exam?",
            placeholder=(
                "e.g. 'I've been really anxious about the exam tomorrow. "
                "I can't sleep and keep forgetting things I studied...'"
            ),
            height=180,
            label_visibility="collapsed",
        )

        char_count = len(user_text)
        st.caption(f"{'✅' if 10 <= char_count <= 1000 else '⚠️'} {char_count}/1000 characters")

        analyze_btn = st.button("🔍 Analyze My Anxiety Level", use_container_width=True)

        st.markdown("""
        <div class="disclaimer-box">
            ⚠️ <strong>Non-Diagnostic Disclaimer:</strong> This tool is intended purely for
            awareness and supportive guidance. It does not provide clinical diagnosis.
            If you experience severe distress, please consult a qualified mental health professional.
        </div>
        """, unsafe_allow_html=True)

    with col_result:
        if analyze_btn:
            if len(user_text.strip()) < 10:
                st.error("Please enter at least 10 characters to analyze.")
            elif len(user_text) > 1000:
                st.error("Please keep your text under 1000 characters.")
            else:
                with st.spinner("Analyzing your text..."):
                    time.sleep(0.5)
                    result = predictor.predict(user_text)

                # Save to history
                st.session_state.history.append({
                    "text": user_text[:80] + ("..." if len(user_text) > 80 else ""),
                    "label": result["label"],
                    "confidence": result["confidence"],
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                })

                label = result["label"]
                color = result["color"]
                emoji = result["emoji"]

                st.markdown(f"""
                <div class="result-card" style="border-color: {color}33;">
                    <span class="result-emoji">{emoji}</span>
                    <div class="result-label" style="color: {color};">{label}</div>
                    <div class="confidence-badge">Confidence: {result['confidence']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

                # Gauge + probability bars
                g_col, p_col = st.columns(2)
                with g_col:
                    st.plotly_chart(render_gauge(result["confidence"], label), use_container_width=True)
                with p_col:
                    st.markdown("##### Probability Breakdown")
                    st.plotly_chart(render_probability_bar(result["probabilities"]), use_container_width=True)

                # Tips
                st.markdown("#### 💡 Personalized Tips")
                tips_html = "".join(f'<div class="tip-item">{tip}</div>' for tip in result["tips"])
                st.markdown(f'<div class="tips-container">{tips_html}</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="height: 200px; display: flex; align-items: center; justify-content: center;
                        color: #4b5563; text-align: center; padding: 2rem;">
                <div>
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📝</div>
                    <div style="font-size: 1rem;">Enter your thoughts and click <strong style="color: #9ca3af;">Analyze</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: BATCH ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif "Batch" in page:
    st.markdown("## 📊 Batch Analysis — Institutional Monitoring")
    st.caption("Analyze multiple student responses at once. All data is processed anonymously.")

    batch_input = st.text_area(
        "Enter one student response per line:",
        placeholder=(
            "I'm very nervous and can't focus at all...\n"
            "Feeling okay, studied enough.\n"
            "Terrified. Can't sleep. My mind is blank.\n"
        ),
        height=200,
    )

    if st.button("📊 Analyze Batch", use_container_width=True):
        lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
        if not lines:
            st.error("Please enter at least one student response.")
        elif len(lines) > 50:
            st.error("Maximum 50 responses per batch.")
        else:
            with st.spinner(f"Analyzing {len(lines)} responses..."):
                results = predictor.predict_batch(lines)
                st.session_state.batch_results = results

            distribution = {"Low Anxiety": 0, "Moderate Anxiety": 0, "High Anxiety": 0}
            for r in results:
                distribution[r["label"]] += 1

            st.markdown("---")
            st.markdown("### 📈 Batch Summary")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Analyzed", len(results))
            with m2:
                st.metric("🟢 Low Anxiety", distribution["Low Anxiety"])
            with m3:
                st.metric("🟡 Moderate", distribution["Moderate Anxiety"])
            with m4:
                st.metric("🔴 High Anxiety", distribution["High Anxiety"])

            pie_col, table_col = st.columns([1, 1])
            with pie_col:
                st.markdown("#### Distribution")
                st.plotly_chart(render_distribution_pie(distribution), use_container_width=True)

            with table_col:
                st.markdown("#### Individual Results")
                rows = []
                for i, (line, r) in enumerate(zip(lines, results), 1):
                    rows.append({
                        "#": i,
                        "Snippet": line[:50] + "..." if len(line) > 50 else line,
                        "Label": f"{r['emoji']} {r['label']}",
                        "Confidence": f"{r['confidence']:.1%}",
                    })
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

            if distribution["High Anxiety"] > 0:
                ratio = distribution["High Anxiety"] / len(results)
                if ratio >= 0.3:
                    st.warning(
                        f"⚠️ **Attention Required**: {distribution['High Anxiety']} students "
                        f"({ratio:.0%}) show signs of high anxiety. Consider organizing counseling sessions or support workshops."
                    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HISTORY
# ─────────────────────────────────────────────────────────────────────────────
elif "History" in page:
    st.markdown("## 📈 Session History & Trends")

    if not st.session_state.history:
        st.info("No analyses performed yet in this session. Go to Single Analysis to get started.")
    else:
        history = st.session_state.history
        df = pd.DataFrame(history)

        dist = df["label"].value_counts().to_dict()
        st.markdown("### This Session")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Analyses Run", len(df))
        c2.metric("🟢 Low", dist.get("Low Anxiety", 0))
        c3.metric("🟡 Moderate", dist.get("Moderate Anxiety", 0))
        c4.metric("🔴 High", dist.get("High Anxiety", 0))

        st.markdown("---")

        # Trend chart
        if len(df) > 1:
            label_order = {"Low Anxiety": 0, "Moderate Anxiety": 1, "High Anxiety": 2}
            df["severity"] = df["label"].map(label_order)
            df["index"] = range(1, len(df) + 1)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["index"], y=df["severity"],
                mode="lines+markers",
                line=dict(color="#818cf8", width=2),
                marker=dict(
                    size=10, color=[ANXIETY_COLORS[l] for l in df["label"]],
                    line=dict(width=2, color="#0d1117"),
                ),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=220,
                margin=dict(t=20, b=20, l=20, r=20),
                yaxis=dict(
                    tickvals=[0, 1, 2],
                    ticktext=["Low", "Moderate", "High"],
                    color="white", gridcolor="rgba(255,255,255,0.05)",
                ),
                xaxis=dict(
                    title="Analysis #", color="#9ca3af",
                    gridcolor="rgba(255,255,255,0.05)",
                ),
            )
            st.markdown("#### Anxiety Trend")
            st.plotly_chart(fig, use_container_width=True)

        # History list
        st.markdown("#### Recent Analyses")
        for item in reversed(history[-10:]):
            color = ANXIETY_COLORS[item["label"]]
            emoji = ANXIETY_EMOJIS[item["label"]]
            st.markdown(f"""
            <div class="history-item" style="border-left-color: {color};">
                <strong style="color: {color};">{emoji} {item['label']}</strong>
                &nbsp;&nbsp;<span style="color: #6b7280; font-size: 0.8rem;">{item['timestamp']} · {item['confidence']:.1%}</span><br>
                <span style="color: #9ca3af;">{item['text']}</span>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
