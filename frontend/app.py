"""
app.py — V5 Fraud Intelligence Dashboard
═════════════════════════════════════════
Premium multi-page Streamlit dashboard for real-time
fraud detection and analysis.

Run: streamlit run app.py --server.port 8501
"""
import streamlit as st
import sys
import os
import textwrap

# Ensure frontend directory is in path for imports
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
if FRONTEND_DIR not in sys.path:
    sys.path.insert(0, FRONTEND_DIR)

from styles import get_main_css

# ══════════════════════════════════
# PAGE CONFIG (must be first)
# ══════════════════════════════════
st.set_page_config(
    page_title="Netra Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════
# INJECT GLOBAL CSS
# ══════════════════════════════════
st.markdown(get_main_css(), unsafe_allow_html=True)

# ══════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════
with st.sidebar:
    # ── Logo / Brand ──
    st.markdown("""
<div style="
text-align: center;
padding: 20px 0 24px;
border-bottom: 1px solid rgba(255, 255, 255, 0.08);
margin-bottom: 24px;
">
<div style="
width: 64px; height: 64px;
background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
border-radius: 16px;
display: flex; align-items: center; justify-content: center;
font-size: 2rem;
margin: 0 auto 12px;
box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
">🛡️</div>
<div style="
font-size: 1.1rem;
font-weight: 800;
background: linear-gradient(135deg, #c7d2fe, #e9d5ff);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
letter-spacing: -0.02em;
">NETRA FRAUD INTEL</div>
<div style="
font-size: 0.65rem;
color: #64748b;
text-transform: uppercase;
letter-spacing: 0.15em;
margin-top: 4px;
">Hybrid Deep Learning Engine</div>
</div>
""", unsafe_allow_html=True)

    # ── Navigation ──
    st.markdown("""
<div style="
color: #64748b;
font-size: 0.65rem;
text-transform: uppercase;
letter-spacing: 0.12em;
padding: 0 16px;
margin-bottom: 8px;
font-weight: 600;
">Navigation</div>
""", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "📊 Fraud Analysis", "🔍 Scan Transactions"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
    st.divider()

    # ── V5 Engine Info ──
    st.markdown("""
<div style="
background: rgba(99, 102, 241, 0.08);
border: 1px solid rgba(99, 102, 241, 0.15);
border-radius: 10px;
padding: 14px 16px;
margin-top: 8px;
">
<div style="
color: #c7d2fe;
font-weight: 700;
font-size: 0.8rem;
margin-bottom: 8px;
">⚡ Netra Hybrid Engine</div>

<div style="font-size: 0.75rem; color: #94a3b8; line-height: 1.6;">
<div style="display: flex; gap: 8px; align-items: center; margin-bottom: 4px;">
<span style="color: #ef4444;">●</span>
<span><b>Path A</b> — XGB+RF (Known)</span>
</div>
<div style="display: flex; gap: 8px; align-items: center; margin-bottom: 4px;">
<span style="color: #f97316;">●</span>
<span><b>Path B</b> — BiLSTM+Attn (Novel)</span>
</div>
<div style="display: flex; gap: 8px; align-items: center; margin-bottom: 4px;">
<span style="color: #f59e0b;">●</span>
<span><b>AE+IF</b> — Anomaly Flags</span>
</div>
<div style="display: flex; gap: 8px; align-items: center;">
<span style="color: #10b981;">●</span>
<span><b>Result</b> — 4-Tier Decision</span>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    # ── Decision Legend ──
    st.markdown("""
<div style="margin-top: 16px; padding: 14px 16px;">
<div style="
color: #64748b;
font-size: 0.65rem;
text-transform: uppercase;
letter-spacing: 0.12em;
margin-bottom: 10px;
font-weight: 600;
">Decision Tiers</div>

<div style="font-size: 0.75rem; line-height: 1.8;">
<div>🔴 <b style="color: #fca5a5;">BLOCK</b>
<span style="color: #64748b;">— Known fraud auto-blocked</span></div>
<div>🟠 <b style="color: #fdba74;">BLOCK_NOVEL</b>
<span style="color: #64748b;">— Novel fraud detected</span></div>
<div>🟡 <b style="color: #fcd34d;">REVIEW</b>
<span style="color: #64748b;">— Flagged for manual review</span></div>
<div>🟢 <b style="color: #6ee7b7;">ALLOW</b>
<span style="color: #64748b;">— Legitimate transaction</span></div>
</div>
</div>
""", unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("""
<div style="
margin-top: 24px;
text-align: center;
color: #475569;
font-size: 0.65rem;
padding-bottom: 16px;
">
Netra Fraud Detection System<br/>
UPI-Based • ML-Powered • Real-Time
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════
# PAGE ROUTING
# ══════════════════════════════════
if page == "🏠 Dashboard":
    from views.dashboard import render_dashboard
    render_dashboard()

elif page == "📊 Fraud Analysis":
    from views.fraud_analysis import render_fraud_analysis
    render_fraud_analysis()

elif page == "🔍 Scan Transactions":
    from views.scan import render_scan
    render_scan()