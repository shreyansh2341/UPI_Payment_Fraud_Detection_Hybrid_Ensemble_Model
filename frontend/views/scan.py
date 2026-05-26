"""
scan.py — Transaction Scanner Page
═══════════════════════════════════
CSV upload and V5 hybrid batch fraud analysis.
4-tier color-coded results with detailed drill-down.
"""
import streamlit as st
import pandas as pd
import time

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    api_predict_batch, api_health_check,
    build_confidence_timeline, build_decision_donut,
    format_number, format_percentage,
)
from styles import metric_card_html, decision_badge


def render_scan():
    """Render the Scan Transactions page."""

    # ── Header ──
    st.markdown("""
<div style="display: flex; align-items: center; gap: 16px; margin-bottom: 8px;">
    <div style="
        width: 48px; height: 48px;
        background: linear-gradient(135deg, #06b6d4, #0891b2);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem;
    ">🔍</div>
    <div>
        <h1 style="margin: 0; padding: 0; font-size: 1.8rem;">Scan Transactions</h1>
        <p style="margin: 0; color: #64748b; font-size: 0.85rem;">
            Upload CSV files for Netra hybrid fraud analysis
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Backend Check ──
    if not api_health_check():
        st.error("⚠️ Backend is offline. Start it with: `uvicorn backend.app:app --port 8003`")
        return

    # ── Configuration ──
    col_config1, col_config2 = st.columns([1, 1])

    with col_config1:
        dataset_type = st.selectbox(
            "🎯 Dataset Logic",
            ["paysim", "creditcard"],
            help="Determines preprocessing and model stack. PaySim uses V5 hybrid, CreditCard uses V3 ensemble.",
        )

    with col_config2:
        st.markdown("""
<div class="info-banner" style="margin-top: 24px;">
    <strong>Netra Hybrid Engine</strong> — Uses V3's XGB+RF for known fraud
    and V4's BiLSTM+Attention for novel fraud detection.
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    # ── Model Stack Info ──
    st.markdown("""
<div style="
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
">
    <div style="
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.15);
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    ">
        <div style="font-size: 1.2rem; margin-bottom: 4px;">📡</div>
        <div style="color: #fca5a5; font-weight: 600; font-size: 0.8rem;">XGBoost + RF</div>
        <div style="color: #64748b; font-size: 0.7rem;">Known Fraud</div>
    </div>
    <div style="
        background: rgba(249, 115, 22, 0.08);
        border: 1px solid rgba(249, 115, 22, 0.15);
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    ">
        <div style="font-size: 1.2rem; margin-bottom: 4px;">🧠</div>
        <div style="color: #fdba74; font-weight: 600; font-size: 0.8rem;">BiLSTM + Attention</div>
        <div style="color: #64748b; font-size: 0.7rem;">Novel Fraud</div>
    </div>
    <div style="
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.15);
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    ">
        <div style="font-size: 1.2rem; margin-bottom: 4px;">🔬</div>
        <div style="color: #fcd34d; font-weight: 600; font-size: 0.8rem;">Autoencoder</div>
        <div style="color: #64748b; font-size: 0.7rem;">Anomaly Detection</div>
    </div>
    <div style="
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.15);
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    ">
        <div style="font-size: 1.2rem; margin-bottom: 4px;">🌲</div>
        <div style="color: #6ee7b7; font-weight: 600; font-size: 0.8rem;">Isolation Forest</div>
        <div style="color: #64748b; font-size: 0.7rem;">Outlier Analysis</div>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── File Upload ──
    uploaded_file = st.file_uploader(
        "📂 Upload Transaction CSV",
        type=["csv"],
        help="Upload raw transaction data. Netra will auto-detect the format and engineer features.",
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # ── Preview ──
        with st.expander("📄 Raw Data Preview", expanded=True):
            st.markdown(f"""
<div style="
    display: flex; gap: 16px; margin-bottom: 12px; flex-wrap: wrap;
">
    <div style="color: #94a3b8; font-size: 0.85rem;">
        <strong style="color: #f1f5f9;">Rows:</strong> {len(df):,}
    </div>
    <div style="color: #94a3b8; font-size: 0.85rem;">
        <strong style="color: #f1f5f9;">Columns:</strong> {len(df.columns)}
    </div>
    <div style="color: #94a3b8; font-size: 0.85rem;">
        <strong style="color: #f1f5f9;">File:</strong> {uploaded_file.name}
    </div>
</div>
""", unsafe_allow_html=True)

            st.dataframe(df.head(10), use_container_width=True)

        # ── Run Analysis ──
        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

        if st.button("🚀 Start Netra Fraud Analysis", use_container_width=True):
            _run_analysis(df, dataset_type, uploaded_file.name)


def _run_analysis(df: pd.DataFrame, dataset_type: str, filename: str):
    """Execute batch analysis and display results."""

    # ── Progress Animation ──
    progress_container = st.empty()
    status_container = st.empty()

    progress_container.markdown("""
<div style="
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 24px;
    text-align: center;
">
    <div style="font-size: 2rem; margin-bottom: 8px;">⚡</div>
    <div style="color: #c7d2fe; font-weight: 600; font-size: 1rem;">
        Running Netra Hybrid Analysis...
    </div>
    <div style="color: #64748b; font-size: 0.85rem; margin-top: 4px;">
        Processing {total} transactions through 4 model layers
    </div>
</div>
""".format(total=f"{len(df):,}"), unsafe_allow_html=True)

    progress_bar = st.progress(0)
    start_time = time.time()

    # Simulate initial progress
    for i in range(30):
        progress_bar.progress(i / 100)
        time.sleep(0.02)

    # ── API Call ──
    result = api_predict_batch(df, dataset_type, filename)

    # Complete progress
    for i in range(30, 101):
        progress_bar.progress(i / 100)
        time.sleep(0.01)

    elapsed = time.time() - start_time

    # Clear progress
    progress_container.empty()
    progress_bar.empty()

    # ── Handle Errors ──
    if "error" in result:
        st.error(f"❌ Analysis Failed: {result['error']}")
        return

    # ── Extract Results ──
    summary = result.get("summary", {})
    results = result.get("results", [])
    session_id = result.get("session_id", "—")

    total = summary.get("total", 0)
    blocked = summary.get("blocked", 0)
    novel = summary.get("novel_blocked", 0)
    review_count = summary.get("review_flagged", 0)
    legit = summary.get("legitimate", 0)
    processing_ms = summary.get("processing_time_ms", 0)

    # ── Success Banner ──
    st.markdown(f"""
<div style="
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(6, 182, 212, 0.1));
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
">
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 1.5rem;">✅</span>
        <div>
            <div style="color: #6ee7b7; font-weight: 700; font-size: 1rem;">
                Analysis Complete
            </div>
            <div style="color: #94a3b8; font-size: 0.8rem;">
                Session: {session_id} • {total:,} transactions in {elapsed:.1f}s
                ({processing_ms:.0f}ms engine time)
            </div>
        </div>
    </div>
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
        st.markdown(metric_card_html(
            "Known Fraud", format_number(blocked),
            card_type="danger",
            delta=format_percentage(blocked, total),
            icon="🔴"
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card_html(
            "Novel Fraud", format_number(novel),
            card_type="warning",
            delta=format_percentage(novel, total),
            icon="🟠"
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card_html(
            "Review", format_number(review_count),
            card_type="review",
            delta=format_percentage(review_count, total),
            icon="🟡"
        ), unsafe_allow_html=True)
    with c5:
        st.markdown(metric_card_html(
            "Legitimate", format_number(legit),
            card_type="success",
            delta=format_percentage(legit, total),
            icon="🟢"
        ), unsafe_allow_html=True)

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    # ── Tabs: Charts + Table ──
    tab_charts, tab_detail, tab_export = st.tabs(["📊 Visualizations", "📋 Detailed Results", "📥 Export"])

    with tab_charts:
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            decision_counts = {}
            for r in results:
                d = r.get("decision", "ALLOW")
                decision_counts[d] = decision_counts.get(d, 0) + 1

            fig = build_decision_donut(decision_counts, "Netra Decision Breakdown")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col_chart2:
            fig = build_confidence_timeline(results, "Confidence Scores by Transaction")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab_detail:
        # Build detailed results DataFrame
        detail_data = []
        for r in results:
            detail_data.append({
                "#": r.get("index", 0),
                "Decision": r.get("decision", "—"),
                "Confidence": f"{r.get('confidence', 0):.4f}",
                "AE Score": f"{r.get('ae_score', 0):.4f}",
                "Seq Score": f"{r.get('seq_score', 0):.4f}",
                "Explanation": r.get("explanation", ""),
            })

        detail_df = pd.DataFrame(detail_data)

        def color_decision_cell(val):
            colors = {
                "BLOCK": "background-color: rgba(239,68,68,0.25); color: #fca5a5; font-weight: 700",
                "BLOCK_NOVEL": "background-color: rgba(249,115,22,0.25); color: #fdba74; font-weight: 700",
                "REVIEW": "background-color: rgba(245,158,11,0.25); color: #fcd34d; font-weight: 700",
                "ALLOW": "background-color: rgba(16,185,129,0.15); color: #6ee7b7; font-weight: 600",
            }
            return colors.get(val, "")

        st.dataframe(
            detail_df.style.applymap(color_decision_cell, subset=["Decision"]),
            use_container_width=True,
            height=500,
        )

        # Fraud-only filter
        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
        show_fraud = st.checkbox("🔴 Show only fraud/suspicious transactions", value=False)

        if show_fraud:
            fraud_df = detail_df[detail_df["Decision"].isin(["BLOCK", "BLOCK_NOVEL", "REVIEW"])]
            st.markdown(f"**{len(fraud_df)}** fraud/suspicious transactions found:")
            st.dataframe(
                fraud_df.style.applymap(color_decision_cell, subset=["Decision"]),
                use_container_width=True,
                height=400,
            )

    with tab_export:
        st.markdown("""
<div class="glass-card">
    <h3 style="margin-top: 0;">📥 Export Analysis Report</h3>
    <p style="color: #94a3b8; font-size: 0.9rem;">
        Download the complete fraud analysis results as a CSV file
        including all Netra hybrid scores and explanations.
    </p>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

        # Merge original data with results
        export_data = []
        for i, r in enumerate(results):
            row = {}
            if i < len(df):
                for col in df.columns:
                    row[col] = df.iloc[i][col]
            row["V5_Decision"] = r.get("decision", "")
            row["V5_Confidence"] = r.get("confidence", 0)
            row["V5_AE_Score"] = r.get("ae_score", 0)
            row["V5_Seq_Score"] = r.get("seq_score", 0)
            row["V5_Explanation"] = r.get("explanation", "")
            export_data.append(row)

        export_df = pd.DataFrame(export_data)

        csv_bytes = export_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Full Audit Report (CSV)",
            data=csv_bytes,
            file_name=f"v5_fraud_audit_{session_id}_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True,
        )
