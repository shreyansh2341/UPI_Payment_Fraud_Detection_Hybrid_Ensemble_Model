"""
fraud_analysis.py — Fraud Analysis Page
════════════════════════════════════════
Time-filtered deep-dive analytics with interactive Plotly charts.
Filters: 1 Week (default), 1 Month, 3 Months, 1 Year.
"""
import streamlit as st
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    api_get_analytics, api_health_check,
    build_fraud_trend_chart, build_decision_donut,
    build_score_histogram, build_hourly_heatmap,
    format_number, format_percentage, get_timeframe_days,
)
from styles import metric_card_html


def render_fraud_analysis():
    """Render the Fraud Analysis page."""

    # ── Header ──
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 8px;">
        <div style="
            width: 48px; height: 48px;
            background: linear-gradient(135deg, #8b5cf6, #a855f7);
            border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.5rem;
        ">📊</div>
        <div>
            <h1 style="margin: 0; padding: 0; font-size: 1.8rem;">Fraud Analysis</h1>
            <p style="margin: 0; color: #64748b; font-size: 0.85rem;">
                Deep-dive analytics on fraud patterns and detection trends
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Backend Check ──
    if not api_health_check():
        st.error("⚠️ Backend is offline. Start it with: `uvicorn backend.app:app --port 8003`")
        return

    # ── Time Frame Filter ──
    st.markdown("""
    <div style="
        background: rgba(17, 24, 39, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 24px;
    ">
        <span style="color: #64748b; font-size: 0.75rem; text-transform: uppercase;
                     letter-spacing: 0.08em; font-weight: 600;">
            📅 Analysis Time Frame
        </span>
    </div>
    """, unsafe_allow_html=True)

    timeframe = st.radio(
        "Select Time Period",
        ["1 Week", "1 Month", "3 Months", "1 Year"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    days = get_timeframe_days(timeframe)

    # ── Fetch Analytics ──
    data = api_get_analytics(days=days)

    if not data:
        st.info("📭 No analytics data available. Upload and scan a CSV first.")
        return

    totals = data.get("totals", {})
    decision_counts = data.get("decision_counts", {})
    daily_series = data.get("daily_series", [])
    hourly_heatmap = data.get("hourly_heatmap", [])
    score_dist = data.get("score_distribution", [])
    sessions = data.get("sessions", [])

    total = totals.get("total", 0) or 0
    blocked = totals.get("blocked", 0) or 0
    novel = totals.get("novel", 0) or 0
    review_count = totals.get("review", 0) or 0
    allowed = totals.get("allowed", 0) or 0
    avg_conf = totals.get("avg_confidence", 0) or 0
    avg_ae = totals.get("avg_ae_score", 0) or 0
    avg_seq = totals.get("avg_seq_score", 0) or 0

    # ── Summary Metrics ──
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.08));
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
    ">
        <div style="text-align: center;">
            <div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase;
                        letter-spacing: 0.08em;">Period</div>
            <div style="color: #f1f5f9; font-weight: 700; font-size: 1.1rem;">{timeframe}</div>
        </div>
        <div style="text-align: center;">
            <div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase;
                        letter-spacing: 0.08em;">Transactions</div>
            <div style="color: #6366f1; font-weight: 700; font-size: 1.1rem;">{format_number(total)}</div>
        </div>
        <div style="text-align: center;">
            <div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase;
                        letter-spacing: 0.08em;">Fraud Rate</div>
            <div style="color: #ef4444; font-weight: 700; font-size: 1.1rem;">
                {format_percentage(blocked + novel, total)}
            </div>
        </div>
        <div style="text-align: center;">
            <div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase;
                        letter-spacing: 0.08em;">Avg Confidence</div>
            <div style="color: #8b5cf6; font-weight: 700; font-size: 1.1rem;">{avg_conf:.4f}</div>
        </div>
        <div style="text-align: center;">
            <div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase;
                        letter-spacing: 0.08em;">Avg AE Score</div>
            <div style="color: #06b6d4; font-weight: 700; font-size: 1.1rem;">{avg_ae:.4f}</div>
        </div>
        <div style="text-align: center;">
            <div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase;
                        letter-spacing: 0.08em;">Avg Seq Score</div>
            <div style="color: #a855f7; font-weight: 700; font-size: 1.1rem;">{avg_seq:.4f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Chart 1: Fraud Detected Over Time ──
    st.markdown("""
    <div class="section-header">
        <h2 style="font-size: 1.2rem;">📈 Fraud Detected Over Time</h2>
    </div>
    """, unsafe_allow_html=True)

    fig_trend = build_fraud_trend_chart(daily_series, f"Daily Fraud Detections — {timeframe}")
    fig_trend.update_layout(height=400)
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Chart 2 & 3: Breakdown + Heatmap ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="section-header">
            <h2 style="font-size: 1.2rem;">🎯 Decision Breakdown</h2>
        </div>
        """, unsafe_allow_html=True)

        fig_donut = build_decision_donut(decision_counts, f"Decisions — {timeframe}")
        fig_donut.update_layout(height=380)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        st.markdown("""
        <div class="section-header">
            <h2 style="font-size: 1.2rem;">🗓️ Fraud by Hour & Day</h2>
        </div>
        """, unsafe_allow_html=True)

        fig_heatmap = build_hourly_heatmap(hourly_heatmap, f"Fraud Heatmap — {timeframe}")
        fig_heatmap.update_layout(height=380)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # ── Chart 4: Score Distributions ──
    st.markdown("""
    <div class="section-header">
        <h2 style="font-size: 1.2rem;">📊 Anomaly Score Distribution</h2>
    </div>
    """, unsafe_allow_html=True)

    fig_scores = build_score_histogram(score_dist, f"AE & Sequential Scores — {timeframe}")
    fig_scores.update_layout(height=380)
    st.plotly_chart(fig_scores, use_container_width=True)

    # ── Upload History ──
    if sessions:
        st.markdown("""
        <div class="section-header">
            <h2 style="font-size: 1.2rem;">📁 Upload Sessions</h2>
        </div>
        """, unsafe_allow_html=True)

        session_data = []
        for s in sessions:
            session_data.append({
                "Timestamp": s.get("timestamp", "—"),
                "File": s.get("filename", "—"),
                "Type": s.get("dataset_type", "—"),
                "Total": s.get("total_count", 0),
                "🔴 Blocked": s.get("fraud_blocked", 0),
                "🟠 Novel": s.get("novel_blocked", 0),
                "🟡 Review": s.get("review_flagged", 0),
                "🟢 Legit": s.get("legitimate", 0),
            })

        st.dataframe(
            pd.DataFrame(session_data),
            use_container_width=True,
            height=300,
        )
