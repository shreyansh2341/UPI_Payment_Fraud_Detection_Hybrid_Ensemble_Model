"""
app.py — V5 Hybrid Fraud Detection API
═══════════════════════════════════════
FastAPI backend with V5 batch inference, analytics, and history endpoints.
"""
import sys
import os
import time

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    FraudRequest, FraudResponse,
    BatchFraudRequest, BatchFraudResponse, BatchSummary, TransactionResult,
    AnalyticsRequest, AnalyticsResponse,
    DashboardResponse,
)
from backend.inference import run_inference, run_v5_batch_inference
from backend import db


app = FastAPI(
    title="V5 Hybrid Fraud Detection API",
    description=(
        "Three-tier hybrid fraud detection: "
        "V3 XGB+RF (Path A) + V4 BiLSTM+Attention (Path B). "
        "Detects both known and novel fraud patterns."
    ),
    version="5.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════
# LEGACY ENDPOINT (backward compat)
# ══════════════════════════════════
@app.post("/predict", response_model=FraudResponse)
def predict_fraud(request: FraudRequest):
    """Legacy single-transaction prediction (V3/V4)."""
    result = run_inference(request)

    return FraudResponse(
        decision=result["decision_label"],
        explanation=result["explanation"],
        confidence=result["score"],
        review_flag=result["review_flag"],
        ae_anomaly_score=result["ae_score"],
        seq_anomaly_score=result.get("seq_score", 0.0),
    )


# ══════════════════════════════════
# V5 BATCH ENDPOINT
# ══════════════════════════════════
@app.post("/predict/v5/batch", response_model=BatchFraudResponse)
def predict_batch_v5(request: BatchFraudRequest):
    """
    V5 batch CSV processing.
    Accepts entire CSV data and returns per-transaction V5 decisions.
    """
    start_ms = time.time() * 1000

    try:
        results = run_v5_batch_inference(
            csv_data=request.csv_data,
            csv_columns=request.csv_columns,
            dataset_type=request.dataset_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    end_ms = time.time() * 1000

    # Extract amounts for storage
    amount_col_idx = None
    cols_lower = [c.lower().strip() for c in request.csv_columns]
    if "amount" in cols_lower:
        amount_col_idx = cols_lower.index("amount")

    amounts = []
    for row in request.csv_data:
        if amount_col_idx is not None and amount_col_idx < len(row):
            try:
                amounts.append(float(row[amount_col_idx]))
            except (ValueError, TypeError):
                amounts.append(0.0)
        else:
            amounts.append(0.0)

    # Persist to DB
    session_id = db.create_session(
        filename=request.filename or "upload.csv",
        dataset_type=request.dataset_type,
    )
    db.store_batch_results(session_id, results, amounts)

    # Build summary
    counts = {"BLOCK": 0, "BLOCK_NOVEL": 0, "REVIEW": 0, "ALLOW": 0, "ERROR": 0}
    total_confidence = 0.0

    transaction_results = []
    for i, r in enumerate(results):
        label = r.get("decision_label", "ALLOW")
        counts[label] = counts.get(label, 0) + 1
        total_confidence += r.get("score", 0.0)

        transaction_results.append(TransactionResult(
            index=i,
            decision=label,
            confidence=r.get("score", 0.0),
            ae_score=r.get("ae_score", 0.0),
            seq_score=r.get("seq_score", 0.0),
            explanation=r.get("explanation", ""),
        ))

    total = len(results)
    summary = BatchSummary(
        total=total,
        blocked=counts.get("BLOCK", 0),
        novel_blocked=counts.get("BLOCK_NOVEL", 0),
        review_flagged=counts.get("REVIEW", 0),
        legitimate=counts.get("ALLOW", 0),
        errors=counts.get("ERROR", 0),
        avg_confidence=total_confidence / max(total, 1),
        processing_time_ms=end_ms - start_ms,
    )

    return BatchFraudResponse(
        session_id=session_id,
        summary=summary,
        results=transaction_results,
    )


# ══════════════════════════════════
# ANALYTICS ENDPOINT
# ══════════════════════════════════
@app.post("/analytics", response_model=AnalyticsResponse)
def get_analytics(request: AnalyticsRequest):
    """Get fraud analytics for a time window."""
    data = db.get_analytics(days=request.days)
    return AnalyticsResponse(**data)


@app.get("/analytics/{days}", response_model=AnalyticsResponse)
def get_analytics_by_days(days: int = 7):
    """Get fraud analytics for a time window (GET convenience)."""
    data = db.get_analytics(days=days)
    return AnalyticsResponse(**data)


# ══════════════════════════════════
# DASHBOARD ENDPOINT
# ══════════════════════════════════
@app.get("/dashboard", response_model=DashboardResponse)
def get_dashboard():
    """Get dashboard summary with recent detections."""
    summary = db.get_dashboard_summary()
    recent = db.get_recent_detections(limit=50)
    return DashboardResponse(
        all_time=summary.get("all_time", {}),
        last_24h=summary.get("last_24h", {}),
        last_session=summary.get("last_session"),
        recent_detections=recent,
    )


# ══════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════
@app.get("/health")
def health_check():
    return {"status": "ok", "version": "5.0", "engine": "V5 Hybrid"}
