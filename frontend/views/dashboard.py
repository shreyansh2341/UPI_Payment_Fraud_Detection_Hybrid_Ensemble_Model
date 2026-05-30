"""
dashboard.py — Real-Time Dashboard Page
════════════════════════════════════════
Live overview with auto-refresh, summary stats, recent detections,
and mini-charts showing fraud trends at a glance.
"""
import streamlit as st
import pandas as pd
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    api_get_dashboard, api_get_analytics, api_health_check,
    build_decision_donut, build_fraud_trend_chart, build_mini_sparkline,
    format_number, format_percentage, time_ago,
)
from styles import metric_card_html, decision_badge


def render_dashboard():
    """Render the Dashboard page."""

    # ── Header ──
    st.markdown("""
<div style="display: flex; align-items: center; gap: 16px; margin-bottom: 8px;">
    <div style="
        width: 48px; height: 48px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem;
    ">🏠</div>
    <div>
        <h1 style="margin: 0; padding: 0; font-size: 1.8rem;">Command Center</h1>
        <p style="margin: 0; color: #64748b; font-size: 0.85rem;">
            Real-time fraud monitoring & detection overview
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Backend Status ──
    backend_ok = api_health_check()

    if not backend_ok:
        st.markdown("""
<div style="
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 20px;
">
    <div style="font-size: 2rem; margin-bottom: 8px;">⚠️</div>
    <div style="color: #fca5a5; font-weight: 600; font-size: 1.1rem;">
        Backend Offline
    </div>
    <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">
        Start the backend: <code>uvicorn backend.app:app --port 8003</code>
    </div>
</div>
""", unsafe_allow_html=True)
        return

    # ── Fetch Data ──
    dashboard_data = api_get_dashboard()
    analytics_7d = api_get_analytics(days=7)

    all_time = dashboard_data.get("all_time", {})
    last_24h = dashboard_data.get("last_24h", {})
    last_session = dashboard_data.get("last_session")
    recent_detections = dashboard_data.get("recent_detections", [])

    total = all_time.get("total", 0) or 0
    blocked = all_time.get("blocked", 0) or 0
    novel = all_time.get("novel", 0) or 0
    review = all_time.get("review", 0) or 0
    allowed = all_time.get("allowed", 0) or 0
    fraud_total = blocked + novel

    # ── Live Status Bar ──
    now_str = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
<div style="
    display: flex; align-items: center; justify-content: space-between;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 10px;
    padding: 10px 20px;
    margin-bottom: 24px;
">
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 8px; height: 8px; background: #10b981; border-radius: 50%;
                    box-shadow: 0 0 8px #10b981;"></div>
        <span style="color: #6ee7b7; font-weight: 600; font-size: 0.85rem;">
            NETRA HYBRID ENGINE — ACTIVE
        </span>
    </div>
    <span style="color: #64748b; font-size: 0.8rem;">Last refresh: {now_str}</span>
</div>
""", unsafe_allow_html=True)

    # ── Metric Cards ──
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(metric_card_html(
            "Total Scanned", format_number(total),
            card_type="total", icon="📊"
        ), unsafe_allow_html=True)

    with c2:
        delta = format_percentage(blocked, total) if total > 0 else "—"
        st.markdown(metric_card_html(
            "Fraud Blocked", format_number(blocked),
            card_type="danger", delta=delta, icon="🔴"
        ), unsafe_allow_html=True)

    with c3:
        st.markdown(metric_card_html(
            "Novel Fraud", format_number(novel),
            card_type="warning", icon="🟠"
        ), unsafe_allow_html=True)

    with c4:
        st.markdown(metric_card_html(
            "Under Review", format_number(review),
            card_type="review", icon="🟡"
        ), unsafe_allow_html=True)

    with c5:
        st.markdown(metric_card_html(
            "Legitimate", format_number(allowed),
            card_type="success", icon="🟢"
        ), unsafe_allow_html=True)

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    # ── Charts Row ──
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        daily_series = analytics_7d.get("daily_series", [])
        fig = build_fraud_trend_chart(daily_series, "Detections — Last 7 Days")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        decision_counts = analytics_7d.get("decision_counts", {})
        fig = build_decision_donut(decision_counts, "Decision Breakdown")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # ── Last Upload Session ──
    if last_session:
        st.markdown(f"""
<div class="glass-card" style="margin-bottom: 24px;">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <div style="color: #64748b; font-size: 0.75rem; text-transform: uppercase;
                        letter-spacing: 0.08em; font-weight: 600;">Latest Upload Session</div>
            <div style="color: #f1f5f9; font-size: 1.1rem; font-weight: 600; margin-top: 4px;">
                📁 {last_session.get('filename', 'N/A')}
            </div>
        </div>
        <div style="text-align: right;">
            <div style="color: #64748b; font-size: 0.75rem;">
                {time_ago(last_session.get('timestamp', ''))}
            </div>
            <div style="margin-top: 4px; display: flex; gap: 8px; justify-content: flex-end;">
                <span style="color: #ef4444; font-weight: 600;">
                    🔴 {last_session.get('fraud_blocked', 0)}
                </span>
                <span style="color: #f97316; font-weight: 600;">
                    🟠 {last_session.get('novel_blocked', 0)}
                </span>
                <span style="color: #f59e0b; font-weight: 600;">
                    🟡 {last_session.get('review_flagged', 0)}
                </span>
                <span style="color: #10b981; font-weight: 600;">
                    🟢 {last_session.get('legitimate', 0)}
                </span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── Recent Detections Table ──
    st.markdown("""
<div class="section-header">
    <h2 style="font-size: 1.3rem;">🔔 Recent Detections</h2>
</div>
""", unsafe_allow_html=True)

    if recent_detections:
        display_data = []
        for det in recent_detections[:30]:
            display_data.append({
                "Time": time_ago(det.get("timestamp", "")),
                "Decision": det.get("decision", "—"),
                "Amount": f"₹{det.get('amount', 0):,.2f}",
                "Confidence": f"{det.get('confidence', 0):.4f}",
                "AE Score": f"{det.get('ae_score', 0):.4f}",
                "Seq Score": f"{det.get('seq_score', 0):.4f}",
                "Source": det.get("filename", "—"),
            })

        display_df = pd.DataFrame(display_data)

        def color_decision(val):
            colors = {
                "BLOCK": "background-color: rgba(239,68,68,0.2); color: #fca5a5; font-weight: 700",
                "BLOCK_NOVEL": "background-color: rgba(249,115,22,0.2); color: #fdba74; font-weight: 700",
                "REVIEW": "background-color: rgba(245,158,11,0.2); color: #fcd34d; font-weight: 700",
                "ALLOW": "background-color: rgba(16,185,129,0.2); color: #6ee7b7; font-weight: 700",
            }
            return colors.get(val, "")

        st.dataframe(
            display_df.style.map(color_decision, subset=["Decision"]),
            use_container_width=True,
            height=400,
        )
    else:
        st.markdown("""
<div style="
    text-align: center; padding: 40px; color: #64748b;
    background: rgba(17, 24, 39, 0.5);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
">
    <div style="font-size: 2.5rem; margin-bottom: 12px;">📭</div>
    <div style="font-size: 1rem; font-weight: 500;">No detections yet</div>
    <div style="font-size: 0.85rem; margin-top: 4px;">
        Go to <b>Scan Transactions</b> to upload a CSV and start analyzing.
    </div>
</div>
""", unsafe_allow_html=True)
