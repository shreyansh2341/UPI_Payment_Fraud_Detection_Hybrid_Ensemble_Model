"""
schemas.py — V5 Hybrid API Schemas
═══════════════════════════════════
Pydantic models for request/response validation.
Supports V5 batch processing, analytics, and history endpoints.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


# ── Decision Enum ──
class Decision(str, Enum):
    BLOCK = "BLOCK"
    BLOCK_NOVEL = "BLOCK_NOVEL"
    REVIEW = "REVIEW"
    ALLOW = "ALLOW"
    FRAUD = "FRAUD"
    LEGIT = "LEGIT"
    ERROR = "ERROR"


# ══════════════════════════════════
# LEGACY ENDPOINTS (backward compat)
# ══════════════════════════════════
class FraudRequest(BaseModel):
    transaction_type: str  # "paysim" or "creditcard"
    tabular_features: List[float]
    lstm_sequence: Optional[List[List[float]]] = None
    model_version: str = "v3"

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
    dataset_type: str = Field(
        ..., description="'paysim' or 'creditcard'"
    )
    csv_data: List[List[Any]] = Field(
        ..., description="List of rows (each row is a list of values)"
    )
    csv_columns: List[str] = Field(
        ..., description="Column names matching the CSV header"
    )
    filename: Optional[str] = Field(
        None, description="Original filename for tracking"
    )


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
    days: int = Field(7, description="Time window in days (7, 30, 90, 365)")


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
