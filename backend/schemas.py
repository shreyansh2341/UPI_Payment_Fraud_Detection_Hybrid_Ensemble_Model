"""
schemas.py — V5 Hybrid API Schemas (Hardened)
══════════════════════════════════════════════
Pydantic models for request/response validation.
Supports V5 batch processing, analytics, and history endpoints.

Security: All inputs are validated with strict bounds to prevent
resource exhaustion and abuse.
"""
import os
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum


# ── Configurable Limits (via env vars) ──
MAX_BATCH_ROWS = int(os.environ.get("NETRA_MAX_BATCH_ROWS", "50000"))
MAX_COLUMNS = 100
MAX_ANALYTICS_DAYS = 365


# ── Decision Enum ──
class Decision(str, Enum):
    BLOCK = "BLOCK"
    BLOCK_NOVEL = "BLOCK_NOVEL"
    REVIEW = "REVIEW"
    ALLOW = "ALLOW"
    FRAUD = "FRAUD"
    LEGIT = "LEGIT"
    ERROR = "ERROR"


# ── Dataset Type Enum (restricts valid values) ──
class DatasetType(str, Enum):
    PAYSIM = "paysim"
    CREDITCARD = "creditcard"


# ══════════════════════════════════
# LEGACY ENDPOINTS (backward compat)
# ══════════════════════════════════
class FraudRequest(BaseModel):
    transaction_type: DatasetType  # Restricted to "paysim" or "creditcard"
    tabular_features: List[float] = Field(
        ..., max_length=200,
        description="Feature values (max 200 features)"
    )
    lstm_sequence: Optional[List[List[float]]] = None
    model_version: str = Field(
        "v3", pattern=r"^v[0-9]+$",
        description="Model version (e.g., 'v3', 'v4', 'v5')"
    )

class FraudResponse(BaseModel):
    decision: str
    explanation: str
    confidence: float = 0.0
    review_flag: bool = False
    ae_anomaly_score: float = 0.0
    seq_anomaly_score: float = 0.0


# ══════════════════════════════════
# V5 BATCH ENDPOINT
# ══════════════════════════════════
class BatchFraudRequest(BaseModel):
    """Request for V5 batch CSV processing."""
    dataset_type: DatasetType = Field(
        ..., description="'paysim' or 'creditcard'"
    )
    csv_data: List[List[Any]] = Field(
        ..., description="List of rows (each row is a list of values)"
    )
    csv_columns: List[str] = Field(
        ..., max_length=MAX_COLUMNS,
        description=f"Column names matching the CSV header (max {MAX_COLUMNS})"
    )
    filename: Optional[str] = Field(
        None, max_length=255,
        description="Original filename for tracking"
    )

    @field_validator("csv_data")
    @classmethod
    def validate_csv_data_size(cls, v):
        """Reject oversized CSV payloads to prevent OOM."""
        if len(v) > MAX_BATCH_ROWS:
            raise ValueError(
                f"CSV exceeds maximum row limit: {len(v)} rows "
                f"(max {MAX_BATCH_ROWS}). Split into smaller batches."
            )
        if len(v) == 0:
            raise ValueError("CSV data cannot be empty.")
        return v

    @field_validator("filename")
    @classmethod
    def sanitize_filename(cls, v):
        """Strip path separators to prevent path traversal."""
        if v is not None:
            # Remove any path separators — only keep the basename
            v = v.replace("/", "").replace("\\", "").replace("..", "")
            if not v:
                v = "upload.csv"
        return v


class TransactionResult(BaseModel):
    """Single transaction result from V5 inference."""
    index: int
    decision: str
    confidence: float = 0.0
    ae_score: float = 0.0
    seq_score: float = 0.0
    explanation: str = ""


class BatchSummary(BaseModel):
    """Aggregated summary of batch results."""
    total: int = 0
    blocked: int = 0
    novel_blocked: int = 0
    review_flagged: int = 0
    legitimate: int = 0
    errors: int = 0
    avg_confidence: float = 0.0
    processing_time_ms: float = 0.0


class BatchFraudResponse(BaseModel):
    """Response for V5 batch CSV processing."""
    session_id: str
    summary: BatchSummary
    results: List[TransactionResult]


# ══════════════════════════════════
# ANALYTICS / HISTORY ENDPOINTS
# ══════════════════════════════════
class AnalyticsRequest(BaseModel):
    days: int = Field(
        7,
        ge=1,
        le=MAX_ANALYTICS_DAYS,
        description=f"Time window in days (1–{MAX_ANALYTICS_DAYS})"
    )


class AnalyticsResponse(BaseModel):
    decision_counts: Dict[str, int] = {}
    daily_series: List[Dict[str, Any]] = []
    hourly_heatmap: List[Dict[str, Any]] = []
    score_distribution: List[Dict[str, Any]] = []
    sessions: List[Dict[str, Any]] = []
    totals: Dict[str, Any] = {}


class DashboardResponse(BaseModel):
    all_time: Dict[str, Any] = {}
    last_24h: Dict[str, Any] = {}
    last_session: Optional[Dict[str, Any]] = None
    recent_detections: List[Dict[str, Any]] = []
