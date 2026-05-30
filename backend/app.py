"""
app.py — V5 Hybrid Fraud Detection API (Hardened)
══════════════════════════════════════════════════
FastAPI backend with V5 batch inference, analytics, and history endpoints.

Security Hardening:
  - Rate limiting via slowapi (per-IP)
  - Restricted CORS origins (env-configurable)
  - Sanitized error messages (no stack trace leakage)
  - Batch row count capped (via Pydantic validators)
  - Input validation on all endpoints
"""
import sys
import os
import time
import logging

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.schemas import (
    FraudRequest, FraudResponse,
    BatchFraudRequest, BatchFraudResponse, BatchSummary, TransactionResult,
    AnalyticsRequest, AnalyticsResponse,
    DashboardResponse,
)
from backend.inference import run_inference, run_v5_batch_inference
from backend import db


# ══════════════════════════════════
# LOGGING
# ══════════════════════════════════
logger = logging.getLogger("netra.api")
logging.basicConfig(level=logging.INFO)


# ══════════════════════════════════
# RATE LIMITER
# ══════════════════════════════════
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="V5 Hybrid Fraud Detection API",
    description=(
        "Three-tier hybrid fraud detection: "
        "V3 XGB+RF (Path A) + V4 BiLSTM+Attention (Path B). "
        "Detects both known and novel fraud patterns."
    ),
    version="5.0",
)

# Register rate limiter with the app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ══════════════════════════════════
# CORS (restricted, env-configurable)
# ══════════════════════════════════
_default_cors_origins = (
    "http://localhost:8501,"
    "http://localhost:7860,"
    "http://127.0.0.1:8501,"
    "http://127.0.0.1:7860"
)
_cors_origins_str = os.environ.get("NETRA_CORS_ORIGINS", _default_cors_origins)
ALLOWED_ORIGINS = [o.strip() for o in _cors_origins_str.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],       # Only methods we actually use
    allow_headers=["Content-Type", "Accept", "Authorization"],
)


# ══════════════════════════════════
# ERROR HANDLER (sanitize responses)
# ══════════════════════════════════
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return a safe error message.
    Never expose raw stack traces or internal paths to clients."""
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )


# ══════════════════════════════════
# LEGACY ENDPOINT (backward compat)
# ══════════════════════════════════
@app.post("/predict", response_model=FraudResponse)
@limiter.limit("10/minute")
def predict_fraud(payload: FraudRequest, request: Request):
    """Legacy single-transaction prediction (V3/V4)."""
    result = run_inference(payload)

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
@limiter.limit("5/minute")
def predict_batch_v5(payload: BatchFraudRequest, request: Request):
    """
    V5 batch CSV processing.
    Accepts entire CSV data and returns per-transaction V5 decisions.
    Rate limited: 5 requests/minute per IP.
    Row count capped by Pydantic validator (default 50,000).
    """
    start_ms = time.time() * 1000

    try:
        results = run_v5_batch_inference(
            csv_data=payload.csv_data,
            csv_columns=payload.csv_columns,
            dataset_type=payload.dataset_type,
        )
    except ValueError as e:
        # Validation errors are safe to relay
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # Log full error internally, return safe message to client
        logger.error(f"Batch inference failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Fraud analysis failed. Check input format and try again.",
        )

    end_ms = time.time() * 1000

    # Extract amounts for storage
    amount_col_idx = None
    cols_lower = [c.lower().strip() for c in payload.csv_columns]
    if "amount" in cols_lower:
        amount_col_idx = cols_lower.index("amount")

    amounts = []
    for row in payload.csv_data:
        if amount_col_idx is not None and amount_col_idx < len(row):
            try:
                amounts.append(float(row[amount_col_idx]))
            except (ValueError, TypeError):
                amounts.append(0.0)
        else:
            amounts.append(0.0)

    # Persist to DB
    session_id = db.create_session(
        filename=payload.filename or "upload.csv",
        dataset_type=payload.dataset_type,
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
@limiter.limit("30/minute")
def get_analytics(payload: AnalyticsRequest, request: Request):
    """Get fraud analytics for a time window."""
    data = db.get_analytics(days=payload.days)
    return AnalyticsResponse(**data)


@app.get("/analytics/{days}", response_model=AnalyticsResponse)
@limiter.limit("30/minute")
def get_analytics_by_days(request: Request, days: int = 7):
    """Get fraud analytics for a time window (GET convenience)."""
    # Clamp days to safe range
    days = max(1, min(days, 365))
    data = db.get_analytics(days=days)
    return AnalyticsResponse(**data)


# ══════════════════════════════════
# DASHBOARD ENDPOINT
# ══════════════════════════════════
@app.get("/dashboard", response_model=DashboardResponse)
@limiter.limit("30/minute")
def get_dashboard(request: Request):
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
@limiter.limit("60/minute")
def health_check(request: Request):
    return {"status": "ok", "version": "5.0", "engine": "V5 Hybrid"}
