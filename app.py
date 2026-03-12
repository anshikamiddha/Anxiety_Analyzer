"""
app.py
AI-Based Exam Anxiety Detection System — Streamlit Frontend
Run with: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
from anxiety_analyzer import analyze_anxiety

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AnxietyLens · Exam Stress Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #070b14;
    color: #e8eaf0;
}

/* ── GIANT "V" BACKGROUND CHARACTER ── */
.stApp::before {
    content: "V";
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-family: 'Syne', sans-serif;
    font-size: 72vw;
    font-weight: 800;
    color: rgba(99, 102, 241, 0.12);
    pointer-events: none;
    z-index: 0;
    line-height: 1;
    letter-spacing: -0.05em;
    user-select: none;
}

.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0f1f3d 0%, #070b14 50%, #0a0f1e 100%);
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; position: relative; z-index: 1; }

h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}

/* ── HERO HEADER ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}
.hero-header .tagline {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.hero-header .sub {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.hero-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #3b4f7a, #6366f1, #3b4f7a, transparent);
    margin: 1.5rem 0;
}

/* ── GLASS CARD ── */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}
.glass-card-accent { border-left: 3px solid #6366f1; }

/* ── ANXIETY BADGE ── */
.anxiety-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-minimal  { background: rgba(34,197,94,0.15);  color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
.badge-mild     { background: rgba(234,179,8,0.15);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.badge-moderate { background: rgba(249,115,22,0.15); color: #fb923c; border: 1px solid rgba(251,146,60,0.3); }
.badge-high     { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
.badge-severe   { background: rgba(168,85,247,0.15); color: #c084fc; border: 1px solid rgba(192,132,252,0.3); }

/* ── CHIPS ── */
.chip-container { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.chip {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    color: #a5b4fc;
}
.chip-red    { background: rgba(239,68,68,0.1);   border: 1px solid rgba(239,68,68,0.25);   color: #fca5a5; }
.chip-yellow { background: rgba(234,179,8,0.1);   border: 1px solid rgba(234,179,8,0.25);   color: #fde047; }

/* ── AI RESPONSE ── */
.ai-response {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(168,85,247,0.06));
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    font-size: 1rem;
    line-height: 1.7;
    color: #cbd5e1;
    position: relative;
    margin-top: 0.5rem;
}
.ai-response::before {
    content: "🤖";
    position: absolute;
    top: -12px; left: 16px;
    font-size: 1.2rem;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: rgba(7,11,20,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── MODEL INFO PILL ── */
.model-pill {
    background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(168,85,247,0.1));
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0 1rem;
}
.model-pill .model-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    color: #a78bfa;
    letter-spacing: 0.03em;
}
.model-pill .model-sub {
    font-size: 0.7rem;
    color: #475569;
    margin-top: 0.15rem;
}

/* ── TEXT AREA ── */
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    resize: vertical;
}
.stTextArea > div > div > textarea:focus {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.1) !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.35) !important;
}

/* ── SECTION LABELS ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #475569;
    margin-bottom: 0.5rem;
}

/* ── RISK FLAG ── */
.risk-flag {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    color: #fca5a5;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}
.risk-flag strong { color: #f87171; }

/* ── FOOTER CREDIT ── */
.footer-credit {
    text-align: center;
    padding: 1.5rem 0 2rem;
}
.footer-credit .made-by {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    background: none;
    -webkit-background-clip: initial;
    -webkit-text-fill-color: initial;
    background-clip: initial;
    letter-spacing: 0.04em;
}
.footer-credit .tech-stack {
    color: #334155;
    font-size: 0.72rem;
    margin-top: 0.4rem;
    letter-spacing: 0.05em;
}

/* ── STATUS DOT ── */
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: #4ade80;
    border-radius: 50%;
    margin-right: 5px;
    box-shadow: 0 0 6px #4ade80;
    animation: blink 2s infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ───────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def get_badge_class(level: str) -> str:
    return {
        "Minimal": "badge-minimal",
        "Mild": "badge-mild",
        "Moderate": "badge-moderate",
        "High": "badge-high",
        "Severe": "badge-severe",
    }.get(level, "badge-mild")

def score_to_color(score: int) -> str:
    if score <= 20: return "#4ade80"
    if score <= 40: return "#fbbf24"
    if score <= 60: return "#fb923c"
    if score <= 80: return "#f87171"
    return "#c084fc"

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.3rem; font-weight: 800;
                    background: linear-gradient(135deg, #60a5fa, #a78bfa);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            ⚙️ Configuration
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model display (static, no dropdown)
    st.markdown("<div class='section-label'>Active Model</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='model-pill'>
        <div class='model-name'>🤗 BERT (Fine-tuned)</div>
        <div class='model-sub'>Trained on GoEmotions Dataset · Google Research</div>
    </div>
    """, unsafe_allow_html=True)

    # API Status
    st.markdown("""
    <div style='font-size:0.75rem; color:#475569; margin-bottom:1rem;'>
        <span class='status-dot'></span>
        <span>API key loaded from <code style="color:#6366f1; font-size:0.7rem;">.env</code></span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Session stats
    if st.session_state.history:
        scores = [h["anxiety_score"] for h in st.session_state.history]
        avg_score = sum(scores) / len(scores)
        peak_score = max(scores)

        st.markdown(f"""
        <div class='section-label'>Session Stats</div>
        <div class='glass-card' style='padding: 1rem;'>
            <div style='display:flex; justify-content:space-between; margin-bottom:0.5rem;'>
                <span style='color:#94a3b8; font-size:0.8rem;'>Queries</span>
                <span style='color:#a78bfa; font-weight:700;'>{len(st.session_state.history)}</span>
            </div>
            <div style='display:flex; justify-content:space-between; margin-bottom:0.5rem;'>
                <span style='color:#94a3b8; font-size:0.8rem;'>Avg Anxiety</span>
                <span style='color:{score_to_color(int(avg_score))}; font-weight:700;'>{avg_score:.1f}</span>
            </div>
            <div style='display:flex; justify-content:space-between;'>
                <span style='color:#94a3b8; font-size:0.8rem;'>Peak Score</span>
                <span style='color:{score_to_color(peak_score)}; font-weight:700;'>{peak_score}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.session_state.current_result = None
            st.rerun()

    st.markdown("---")

    # History list
    if st.session_state.history:
        st.markdown("<div class='section-label'>Recent Queries</div>", unsafe_allow_html=True)
        for i, entry in enumerate(reversed(st.session_state.history[-8:])):
            idx = len(st.session_state.history) - 1 - i
            preview = entry["raw_input"][:50] + "..." if len(entry["raw_input"]) > 50 else entry["raw_input"]
            if st.button(f"[{entry['anxiety_score']}] {preview}", key=f"hist_{idx}"):
                st.session_state.current_result = entry

# ─── MAIN CONTENT ────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-header'>
    <div class='tagline'>AnxietyLens</div>
    <div class='sub'>AI-Powered Exam Stress Detection System</div>
</div>
<div class='hero-divider'></div>
""", unsafe_allow_html=True)

# ─── INPUT SECTION ───────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("<div class='section-label'>Tell us how you're feeling about your exams</div>", unsafe_allow_html=True)

    user_input = st.text_area(
        "",
        height=160,
        placeholder="e.g. 'I have my finals in 3 days and I haven't studied enough. I keep blanking on everything I read and I can't sleep. What if I fail and disappoint everyone?'",
        label_visibility="collapsed"
    )

    st.markdown("<div class='section-label' style='margin-top:0.5rem;'>Try an example</div>", unsafe_allow_html=True)
    examples = [
        "I'm a bit nervous about tomorrow's test but I've been studying well.",
        "I can't stop shaking. I haven't slept in 2 days. Everything I know goes blank during tests.",
        "My finals are in a week. I'm managing okay, but I get worried about running out of time.",
    ]
    for ex in examples:
        if st.button(f"▸ {ex[:55]}...", key=ex[:20]):
            st.session_state["prefill"] = ex
            st.rerun()

    if "prefill" in st.session_state:
        user_input = st.session_state["prefill"]
        del st.session_state["prefill"]

    analyze_btn = st.button("🔬 Analyze Anxiety", key="analyze")

# ─── ANALYSIS TRIGGER ─── (no API key needed — loaded from .env automatically)
if analyze_btn:
    if not user_input.strip():
        st.warning("Please type something about how you're feeling.")
    else:
        with st.spinner("🧠 Analyzing emotional patterns..."):
            result = analyze_anxiety(user_input)   # key comes from .env via anxiety_analyzer.py
            result["timestamp"] = datetime.now().strftime("%H:%M:%S")
            st.session_state.history.append(result)
            st.session_state.current_result = result
            st.rerun()

# ─── RESULTS PANEL ───────────────────────────────────────────────────────────────
result = st.session_state.current_result

with col_result:
    if result:
        score = result.get("anxiety_score", 0)
        level = result.get("anxiety_level", "Unknown")
        tone  = result.get("emotional_tone", "")
        badge_class = get_badge_class(level)
        color = score_to_color(score)

        st.markdown(f"""
        <div class='glass-card glass-card-accent'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;'>
                <span class='anxiety-badge {badge_class}'>{level}</span>
                <span style='color:#64748b; font-size:0.75rem;'>{result.get("timestamp", "")}</span>
            </div>
            <div style='display:flex; align-items:baseline; gap:0.5rem; margin-bottom:0.3rem;'>
                <span style='font-family: Syne, sans-serif; font-size: 3.5rem; font-weight: 800; color: {color}; line-height:1;'>{score}</span>
                <span style='color:#475569; font-size:1rem;'>/100</span>
                <span style='color:#94a3b8; font-size:0.9rem; margin-left:0.5rem;'>· {tone}</span>
            </div>
            <div style='height:8px; background:rgba(255,255,255,0.06); border-radius:999px; overflow:hidden;'>
                <div style='height:100%; width:{score}%; background: linear-gradient(90deg, #4ade80, {color}); border-radius:999px; transition: width 0.5s ease;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='section-label'>Supportive Response</div>
        <div class='ai-response'>{result.get("supportive_response", "")}</div>
        """, unsafe_allow_html=True)

        risk_flags = result.get("risk_flags", [])
        if risk_flags and any(risk_flags):
            st.markdown(f"""
            <div class='risk-flag'>
                <strong>⚠️ Risk Indicators Detected:</strong> {", ".join(risk_flags)}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='height:300px; display:flex; align-items:center; justify-content:center;
                    flex-direction:column; gap:1rem; opacity:0.4;'>
            <div style='font-size:3rem;'>🧠</div>
            <div style='font-family: Syne, sans-serif; color:#475569; text-align:center;'>
                Your analysis will appear here.<br>Type your thoughts and click Analyze.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── DETAILED BREAKDOWN ──────────────────────────────────────────────────────────
if result:
    st.markdown("<div class='hero-divider' style='margin:1.5rem 0;'></div>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3, gap="medium")

    with col_a:
        st.markdown("<div class='section-label'>Key Indicators</div>", unsafe_allow_html=True)
        chips = "".join([f"<span class='chip'>{i}</span>" for i in result.get("key_indicators", [])])
        st.markdown(f"<div class='chip-container'>{chips}</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='section-label'>Cognitive Distortions</div>", unsafe_allow_html=True)
        dist = result.get("cognitive_distortions", [])
        if dist:
            chips = "".join([f"<span class='chip chip-yellow'>{i}</span>" for i in dist])
            st.markdown(f"<div class='chip-container'>{chips}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#475569; font-size:0.8rem;'>None detected ✓</span>", unsafe_allow_html=True)

    with col_c:
        st.markdown("<div class='section-label'>Physical Symptoms</div>", unsafe_allow_html=True)
        syms = result.get("physical_symptoms_mentioned", [])
        if syms:
            chips = "".join([f"<span class='chip chip-red'>{i}</span>" for i in syms])
            st.markdown(f"<div class='chip-container'>{chips}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#475569; font-size:0.8rem;'>None mentioned</span>", unsafe_allow_html=True)

    # Coping Strategies
    st.markdown("<div class='hero-divider' style='margin:1.2rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Recommended Coping Strategies</div>", unsafe_allow_html=True)
    strategies = result.get("coping_strategies", [])
    if strategies:
        cols = st.columns(len(strategies))
        icons = ["🧘", "📝", "💬", "🏃", "😴"]
        for i, (col, strat) in enumerate(zip(cols, strategies)):
            with col:
                st.markdown(f"""
                <div class='glass-card' style='text-align:center; padding:1rem;'>
                    <div style='font-size:1.5rem; margin-bottom:0.4rem;'>{icons[i % len(icons)]}</div>
                    <div style='font-size:0.82rem; color:#94a3b8;'>{strat}</div>
                </div>
                """, unsafe_allow_html=True)

# ─── ANALYTICS DASHBOARD ─────────────────────────────────────────────────────────
if len(st.session_state.history) >= 1:
    st.markdown("<div class='hero-divider' style='margin:1.5rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family: Syne, sans-serif; font-size: 1.4rem; font-weight: 700;
                color:#e2e8f0; margin-bottom: 1rem;'>
        📊 Analytics Dashboard
    </div>
    """, unsafe_allow_html=True)

    df = pd.DataFrame(st.session_state.history)
    df["index"] = range(1, len(df) + 1)
    df["preview"] = df["raw_input"].apply(lambda x: x[:30] + "..." if len(x) > 30 else x)

    chart_col1, chart_col2 = st.columns([3, 2], gap="large")

    with chart_col1:
        fig_line = go.Figure()
        colors = [score_to_color(s) for s in df["anxiety_score"]]

        fig_line.add_trace(go.Scatter(
            x=df["index"], y=df["anxiety_score"],
            fill='tozeroy', fillcolor='rgba(99,102,241,0.06)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False, hoverinfo='skip'
        ))
        fig_line.add_trace(go.Scatter(
            x=df["index"], y=df["anxiety_score"],
            mode='lines+markers',
            line=dict(color='#818cf8', width=2.5, shape='spline'),
            marker=dict(size=10, color=colors, line=dict(width=2, color='rgba(0,0,0,0.5)')),
            text=df["anxiety_level"],
            customdata=df["preview"],
            hovertemplate='<b>Query %{x}</b><br>Score: %{y}<br>Level: %{text}<br><i>%{customdata}</i><extra></extra>',
            showlegend=False
        ))
        for y_val, label, c in [(20,'Minimal','rgba(74,222,128,0.05)'),
                                  (40,'Mild','rgba(251,191,36,0.05)'),
                                  (60,'Moderate','rgba(251,146,60,0.05)'),
                                  (80,'High','rgba(248,113,113,0.05)')]:
            fig_line.add_hrect(y0=y_val-20, y1=y_val, fillcolor=c, line_width=0,
                               annotation_text=label, annotation_position="right",
                               annotation_font=dict(color='#475569', size=9))
        fig_line.update_layout(
            title=dict(text="Anxiety Score Timeline", font=dict(family='Syne', size=14, color='#94a3b8')),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, color='#475569', tickfont=dict(size=10), title="Query #"),
            yaxis=dict(range=[0,105], showgrid=True, gridcolor='rgba(255,255,255,0.04)',
                       color='#475569', tickfont=dict(size=10), title="Score"),
            margin=dict(l=0, r=60, t=40, b=0), height=280,
            font=dict(family='DM Sans'),
            hoverlabel=dict(bgcolor='#1e293b', bordercolor='#334155', font=dict(color='#e2e8f0'))
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with chart_col2:
        current_score = st.session_state.current_result["anxiety_score"] if st.session_state.current_result else 0
        current_color = score_to_color(current_score)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_score,
            delta={'reference': df["anxiety_score"].mean(), 'relative': False,
                   'font': {'size': 12, 'color': '#94a3b8'}},
            title={'text': "Current Score", 'font': {'size': 13, 'color': '#94a3b8', 'family': 'Syne'}},
            number={'font': {'size': 40, 'color': current_color, 'family': 'Syne'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#334155', 'tickfont': {'size': 9, 'color':'#475569'}},
                'bar': {'color': current_color, 'thickness': 0.25},
                'bgcolor': 'rgba(0,0,0,0)', 'borderwidth': 0,
                'steps': [
                    {'range': [0,  20], 'color': 'rgba(74,222,128,0.08)'},
                    {'range': [20, 40], 'color': 'rgba(251,191,36,0.08)'},
                    {'range': [40, 60], 'color': 'rgba(251,146,60,0.08)'},
                    {'range': [60, 80], 'color': 'rgba(248,113,113,0.08)'},
                    {'range': [80,100], 'color': 'rgba(192,132,252,0.08)'},
                ],
                'threshold': {'line': {'color': current_color, 'width': 3}, 'value': current_score}
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', height=280,
            margin=dict(l=20, r=20, t=40, b=0), font=dict(family='DM Sans')
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    if len(st.session_state.history) >= 2:
        chart_col3, chart_col4 = st.columns(2, gap="large")

        with chart_col3:
            level_counts = df["anxiety_level"].value_counts().reset_index()
            level_counts.columns = ["Level", "Count"]
            level_color_map = {"Minimal":"#4ade80","Mild":"#fbbf24",
                               "Moderate":"#fb923c","High":"#f87171","Severe":"#c084fc"}
            pie_colors = [level_color_map.get(l, "#6366f1") for l in level_counts["Level"]]
            fig_pie = go.Figure(go.Pie(
                labels=level_counts["Level"], values=level_counts["Count"], hole=0.65,
                marker=dict(colors=pie_colors, line=dict(color='#070b14', width=3)),
                textfont=dict(size=11, color='#e2e8f0'),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>'
            ))
            fig_pie.update_layout(
                title=dict(text="Anxiety Level Distribution", font=dict(family='Syne', size=14, color='#94a3b8')),
                paper_bgcolor='rgba(0,0,0,0)', height=280,
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(font=dict(color='#94a3b8', size=10), bgcolor='rgba(0,0,0,0)'),
                annotations=[dict(text=f"<b>{len(df)}</b><br>queries",
                                  x=0.5, y=0.5, font_size=16, font_color='#e2e8f0',
                                  font_family='Syne', showarrow=False)]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col4:
            tone_counts = df["emotional_tone"].value_counts().head(6).reset_index()
            tone_counts.columns = ["Tone", "Count"]
            fig_bar = go.Figure(go.Bar(
                x=tone_counts["Count"], y=tone_counts["Tone"], orientation='h',
                marker=dict(color=tone_counts["Count"],
                            colorscale=[[0,'#4f46e5'],[0.5,'#7c3aed'],[1,'#c026d3']],
                            showscale=False, line=dict(width=0)),
                text=tone_counts["Count"], textposition='outside',
                textfont=dict(color='#94a3b8', size=11),
                hovertemplate='<b>%{y}</b>: %{x} times<extra></extra>'
            ))
            fig_bar.update_layout(
                title=dict(text="Emotional Tone Frequency", font=dict(family='Syne', size=14, color='#94a3b8')),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)',
                           color='#475569', tickfont=dict(size=10)),
                yaxis=dict(showgrid=False, color='#e2e8f0', tickfont=dict(size=10)),
                height=280, margin=dict(l=0, r=40, t=40, b=0),
                font=dict(family='DM Sans'),
                hoverlabel=dict(bgcolor='#1e293b', bordercolor='#334155', font=dict(color='#e2e8f0'))
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# ─── FOOTER ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-divider' style='margin:2.5rem 0 0;'></div>
<div class='footer-credit'>
    <div class='made-by'>Made by 🌻team - 6997fc54b750b5acb2d6739a</div>
    <div class='tech-stack'>
        BERT · GoEmotions · FastAPI · Streamlit ·  LLaMA &nbsp;·&nbsp;
        <span style='color:#6366f1;'>AnxietyLens v1.0</span> &nbsp;·&nbsp;
        For educational purposes only. Not a clinical tool.
    </div>
</div>
""", unsafe_allow_html=True)