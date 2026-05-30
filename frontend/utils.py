"""
utils.py — Frontend Utility Functions
══════════════════════════════════════
API client, chart builders, and data formatters
for the V5 Fraud Intelligence Dashboard.
"""
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

from styles import get_plotly_theme

API_BASE = "https://shreyansh2341-netra-advance-ai-to-detect-and-blo-6352edf.hf.space"


# ══════════════════════════════════
# API CLIENT
# ══════════════════════════════════
def api_predict_batch(df: pd.DataFrame, dataset_type: str, filename: str = "upload.csv"):
    """Send batch CSV data to V5 endpoint."""
    csv_data = df.values.tolist()
    csv_columns = df.columns.tolist()

    payload = {
        "dataset_type": dataset_type,
        "csv_data": csv_data,
        "csv_columns": csv_columns,
        "filename": filename,
    }

    try:
        response = requests.post(
            f"{API_BASE}/predict/v5/batch",
            json=payload,
            timeout=300,
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Server error: {response.status_code} — {response.text}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Is it running on port 8003?"}
    except Exception as e:
        return {"error": str(e)}


def api_get_dashboard():
    """Fetch dashboard summary."""
    try:
        r = requests.get(f"{API_BASE}/dashboard", timeout=10)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def api_get_analytics(days: int = 7):
    """Fetch analytics for time window."""
    try:
        r = requests.get(f"{API_BASE}/analytics/{days}", timeout=10)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def api_health_check():
    """Check if backend is running."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ══════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════
def build_fraud_trend_chart(daily_series: list, title: str = "Fraud Detections Over Time"):
    """Line chart showing fraud detections by day and type."""
    if not daily_series:
        return _empty_chart(title)

    df = pd.DataFrame(daily_series)
    theme = get_plotly_theme()

    fig = go.Figure()

    decision_colors = {
        "BLOCK": "#ef4444",
        "BLOCK_NOVEL": "#f97316",
        "REVIEW": "#f59e0b",
        "ALLOW": "#10b981",
    }

    for decision in ["BLOCK", "BLOCK_NOVEL", "REVIEW", "ALLOW"]:
        subset = df[df["decision"] == decision]
        if len(subset) > 0:
            fig.add_trace(go.Scatter(
                x=subset["date"],
                y=subset["cnt"],
                mode="lines+markers",
                name=decision,
                line=dict(
                    color=decision_colors.get(decision, "#6366f1"),
                    width=2.5,
                ),
                marker=dict(size=6),
                fill="tozeroy" if decision in ("BLOCK", "BLOCK_NOVEL") else None,
                fillcolor=f"rgba({','.join(str(int(decision_colors.get(decision, '#6366f1').lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))}, 0.1)" if decision in ("BLOCK", "BLOCK_NOVEL") else None,
                hovertemplate=f"<b>{decision}</b><br>Date: %{{x}}<br>Count: %{{y}}<extra></extra>",
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Detections",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        **theme,
    )

    return fig


def build_decision_donut(decision_counts: dict, title: str = "Decision Breakdown"):
    """Donut chart for decision distribution."""
    if not decision_counts:
        return _empty_chart(title)

    labels = list(decision_counts.keys())
    values = list(decision_counts.values())

    colors = {
        "BLOCK": "#ef4444",
        "BLOCK_NOVEL": "#f97316",
        "REVIEW": "#f59e0b",
        "ALLOW": "#10b981",
    }
    marker_colors = [colors.get(l, "#6366f1") for l in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=marker_colors, line=dict(color="#0a0e1a", width=2)),
        textinfo="percent+label",
        textfont=dict(size=12, family="Inter"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
    )])

    theme = get_plotly_theme()
    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        **theme,
    )

    return fig


def build_score_histogram(scores: list, title: str = "Anomaly Score Distribution"):
    """Histogram of AE/Seq scores."""
    if not scores:
        return _empty_chart(title)

    df = pd.DataFrame(scores)
    theme = get_plotly_theme()

    fig = go.Figure()

    if "ae_score" in df.columns:
        ae_vals = df["ae_score"].dropna()
        if len(ae_vals) > 0:
            fig.add_trace(go.Histogram(
                x=ae_vals,
                name="AE Score",
                marker_color="rgba(99, 102, 241, 0.7)",
                nbinsx=40,
                hovertemplate="AE Score: %{x:.4f}<br>Count: %{y}<extra></extra>",
            ))

    if "seq_score" in df.columns:
        seq_vals = df["seq_score"].dropna()
        if len(seq_vals) > 0:
            fig.add_trace(go.Histogram(
                x=seq_vals,
                name="Seq Score",
                marker_color="rgba(139, 92, 246, 0.7)",
                nbinsx=40,
                hovertemplate="Seq Score: %{x:.4f}<br>Count: %{y}<extra></extra>",
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Score",
        yaxis_title="Count",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **theme,
    )

    return fig


def build_hourly_heatmap(heatmap_data: list, title: str = "Fraud by Hour & Day"):
    """Heatmap of fraud detections by hour and day of week."""
    if not heatmap_data:
        return _empty_chart(title)

    days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    grid = np.zeros((7, 24))

    for entry in heatmap_data:
        dow = int(entry.get("dow", 0))
        hour = int(entry.get("hour", 0))
        cnt = int(entry.get("cnt", 0))
        if 0 <= dow < 7 and 0 <= hour < 24:
            grid[dow][hour] = cnt

    theme = get_plotly_theme()

    fig = go.Figure(data=go.Heatmap(
        z=grid,
        x=[f"{h:02d}:00" for h in range(24)],
        y=days,
        colorscale=[
            [0, "rgba(10, 14, 26, 0.8)"],
            [0.25, "rgba(99, 102, 241, 0.3)"],
            [0.5, "rgba(139, 92, 246, 0.5)"],
            [0.75, "rgba(249, 115, 22, 0.7)"],
            [1, "rgba(239, 68, 68, 0.9)"],
        ],
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Fraud Count: %{z}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        **theme,
    )

    return fig


def build_confidence_timeline(results: list, title: str = "Confidence Score Timeline"):
    """Scatter plot of confidence scores across transactions."""
    if not results:
        return _empty_chart(title)

    df = pd.DataFrame(results)
    theme = get_plotly_theme()

    decision_colors = {
        "BLOCK": "#ef4444",
        "BLOCK_NOVEL": "#f97316",
        "REVIEW": "#f59e0b",
        "ALLOW": "#10b981",
    }

    fig = go.Figure()

    for decision in ["BLOCK", "BLOCK_NOVEL", "REVIEW", "ALLOW"]:
        subset = df[df["decision"] == decision]
        if len(subset) > 0:
            fig.add_trace(go.Scatter(
                x=subset["index"],
                y=subset["confidence"],
                mode="markers",
                name=decision,
                marker=dict(
                    color=decision_colors.get(decision, "#6366f1"),
                    size=5,
                    opacity=0.7,
                ),
                hovertemplate=(
                    f"<b>{decision}</b><br>"
                    "Transaction: %{x}<br>"
                    "Confidence: %{y:.4f}<extra></extra>"
                ),
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Transaction Index",
        yaxis_title="Confidence Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **theme,
    )

    return fig


def build_mini_sparkline(values: list, color: str = "#6366f1"):
    """Simple mini sparkline for dashboard cards."""
    if not values or len(values) < 2:
        return _empty_chart("")

    fig = go.Figure(data=go.Scatter(
        y=values,
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))}, 0.1)",
    ))

    fig.update_layout(
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )

    return fig


def _empty_chart(title: str):
    """Return an empty chart placeholder."""
    theme = get_plotly_theme()
    fig = go.Figure()
    fig.add_annotation(
        text="No data available yet.<br>Upload a CSV to see results.",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="#64748b"),
    )
    fig.update_layout(
        title=title,
        height=300,
        **theme,
    )
    return fig


# ══════════════════════════════════
# DATA FORMATTERS
# ══════════════════════════════════
def format_number(n):
    """Format large numbers with K/M suffixes."""
    if n is None:
        return "0"
    n = int(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def format_percentage(part, total):
    """Format as percentage string."""
    if total == 0:
        return "0%"
    return f"{part/total*100:.1f}%"


def time_ago(timestamp_str: str) -> str:
    """Convert ISO timestamp to human-readable 'time ago'."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        diff = datetime.utcnow() - dt.replace(tzinfo=None)

        if diff.total_seconds() < 60:
            return "just now"
        if diff.total_seconds() < 3600:
            return f"{int(diff.total_seconds() // 60)}m ago"
        if diff.total_seconds() < 86400:
            return f"{int(diff.total_seconds() // 3600)}h ago"
        return f"{diff.days}d ago"
    except Exception:
        return timestamp_str


def get_timeframe_days(timeframe: str) -> int:
    """Convert timeframe label to days."""
    mapping = {
        "1 Week": 7,
        "1 Month": 30,
        "3 Months": 90,
        "1 Year": 365,
    }
    return mapping.get(timeframe, 7)
