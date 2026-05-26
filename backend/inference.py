"""
inference.py — V5 Hybrid Backend Inference (Hardened)
═════════════════════════════════════════════════════
Auto-detects CSV format, engineers velocity features,
then calls the V5 hybrid ensemble (V3 Path A + V4 Path B).

V5 Decision Tiers:
  Tier 1 (BLOCK):         V3's XGB+RF detects known fraud
  Tier 2 (BLOCK_NOVEL):   V4's BiLSTM confirms novel fraud
  Tier 2b (BLOCK_SMURFING): Anti-smurfing velocity detection
  Tier 3 (REVIEW):        Anomaly flags for manual review
  Otherwise (ALLOW):      Transaction is legitimate

Robustness Mitigations (per stress test audit):
  [M1] Noise Guard:     Input sanitization on all entry points
  [M5] Production Mode: Toggle for leaky feature neutralization
  [M6] Robust Clipping: Velocity feature outlier clamping
"""
import sys
import os
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.final_ensemble_inference import ensemble_predict
from src.utils.preprocessor import clean_and_engineer_upi
from src.robustness_mitigations import noise_guard_sanitize

# ── Lazy model caches ──
_V4_MODELS = None
_V5_MODELS = None


def _load_v4_models():
    global _V4_MODELS
    if _V4_MODELS is None:
        from src.model_loader import load_paysim_v4_hybrid
        _V4_MODELS = load_paysim_v4_hybrid()
    return _V4_MODELS


def _load_v5_models():
    """Lazily load V5 (V3+V4) models on first V5 request."""
    global _V5_MODELS
    if _V5_MODELS is None:
        from src.model_loader import load_v5_hybrid
        _V5_MODELS = load_v5_hybrid()
    return _V5_MODELS


# ── Feature Schemas ──
PAYSIM_V3_FEATURES = [
    "amount", "oldbalanceorg", "newbalanceorig", "oldbalancedest",
    "newbalancedest", "hour", "dayofweek", "is_weekend",
    "errorbalanceorig", "errorbalancedest",
    "upi_type_upi_payment", "upi_type_upi_transfer",
    "tx_count_cumul", "amount_cumul", "amt_vs_avg",
    "time_since_last", "amt_to_bal_ratio", "balance_velocity",
]

ENGINEERED_FULL_COLUMNS = [
    "amount", "oldbalanceorg", "newbalanceorig", "oldbalancedest",
    "newbalancedest", "hour", "dayofweek", "is_weekend",
    "errorbalanceorig", "errorbalancedest",
    "upi_type_upi_payment", "upi_type_upi_transfer"
]

RAW_PAYSIM_COLUMNS = [
    "step", "type", "amount",
    "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud"
]

CREDITCARD_COLUMNS = [
    "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
    "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
    "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
    "amount_scaled", "hour", "dayofweek", "is_weekend"
]


def add_velocity_defaults(df):
    """Add velocity features with sensible defaults for single-transaction inference.
    
    [M6] Robust Clipping: All computed ratios are clamped to safe ranges
    to prevent noise-induced extreme values from destabilizing the scaler.
    Per Noise_Injection_Report: unclamped values caused 40% recall drop.
    """
    # Safe extraction with NaN protection
    amount = float(df["amount"].iloc[0]) if not pd.isna(df["amount"].iloc[0]) else 0.0
    old_bal = float(df["oldbalanceorg"].iloc[0]) if not pd.isna(df["oldbalanceorg"].iloc[0]) else 0.0
    new_bal = float(df["newbalanceorig"].iloc[0]) if not pd.isna(df["newbalanceorig"].iloc[0]) else 0.0

    # Compute velocity features with robust clipping
    amt_to_bal = max(0.0, min(amount / (old_bal + 1e-6), 1e6))  # [M6] Cap ratio
    bal_velocity = (new_bal - old_bal) / (amount + 1e-6)
    bal_velocity = max(-100.0, min(bal_velocity, 10.0))  # [M6] Cap velocity

    defaults = {
        "tx_count_cumul": 1.0,
        "amount_cumul": max(0.0, amount),
        "amt_vs_avg": 1.0,
        "time_since_last": 48.0,
        "amt_to_bal_ratio": amt_to_bal,
        "balance_velocity": bal_velocity,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
            
    # CRITICAL FIX: Log-transform heavy-tailed features exactly like V3/V4 training
    if "amount_cumul" in df.columns:
        df["amount_cumul"] = np.log1p(df["amount_cumul"].clip(lower=0))
    if "tx_count_cumul" in df.columns:
        df["tx_count_cumul"] = np.log1p(df["tx_count_cumul"].astype(float))
    if "amt_to_bal_ratio" in df.columns:
        df["amt_to_bal_ratio"] = np.log1p(df["amt_to_bal_ratio"].clip(lower=0))
        
    return df


def add_velocity_defaults_batch(df):
    """Add velocity features for batch processing treating each row independently if missing.
    
    [M6] Robust Clipping: All computed ratios are clamped to prevent noise-induced
    extreme values. Per Noise_Injection_Report: extreme ratios caused precision
    to drop from 99.68% → 1.8% at just 5% noise.
    """
    if "tx_count_cumul" not in df.columns:
        df["tx_count_cumul"] = 1.0
    if "amount_cumul" not in df.columns:
        df["amount_cumul"] = df["amount"].clip(lower=0)  # [M6] No negatives
    if "amt_vs_avg" not in df.columns:
        df["amt_vs_avg"] = 1.0
    if "time_since_last" not in df.columns:
        df["time_since_last"] = 48.0  # Default gap
    if "amt_to_bal_ratio" not in df.columns:
        df["amt_to_bal_ratio"] = (
            (df["amount"] / (df["oldbalanceorg"] + 1e-6)).clip(0, 1e6)  # [M6] Cap
        )
    if "balance_velocity" not in df.columns:
        df["balance_velocity"] = (
            (df["newbalanceorig"] - df["oldbalanceorg"]) / (df["amount"] + 1e-6)
        ).clip(-100, 10)  # [M6] Cap velocity range
        
    # CRITICAL FIX: Log-transform heavy-tailed features exactly like V3/V4 training
    if "amount_cumul" in df.columns:
        df["amount_cumul"] = np.log1p(df["amount_cumul"].clip(lower=0))
    if "tx_count_cumul" in df.columns:
        df["tx_count_cumul"] = np.log1p(df["tx_count_cumul"].astype(float))
    if "amt_to_bal_ratio" in df.columns:
        df["amt_to_bal_ratio"] = np.log1p(df["amt_to_bal_ratio"].clip(lower=0))

    return df


def detect_csv_format(values, num_cols):
    """Detect CSV format: v3_full, engineered_full, engineered_legacy, raw, or creditcard."""
    if num_cols == 18:
        return 'v3_full'
    if num_cols == 13:
        return 'engineered_full'
    if num_cols in (11, 12):
        try:
            if float(values[1]) > 1000:
                return 'engineered_legacy'
        except (ValueError, TypeError):
            pass
    if num_cols >= 30:
        return 'creditcard'
    return 'raw'


def detect_csv_format_by_columns(columns):
    """Detect CSV format by column names (more reliable for batch)."""
    cols_lower = [c.lower().strip() for c in columns]
    num_cols = len(cols_lower)

    # Check for raw PaySim by column names
    raw_indicators = {"step", "type", "nameorig", "namedest"}
    if raw_indicators.issubset(set(cols_lower)):
        return 'raw'

    # Credit card by V1-V28 features
    v_cols = sum(1 for c in cols_lower if c.startswith('v') and c[1:].isdigit())
    if v_cols >= 20:
        return 'creditcard'

    # Check for already engineered features
    engineered_indicators = {"errorbalanceorig", "errorbalancedest", "is_weekend"}
    if engineered_indicators.issubset(set(cols_lower)):
        if "tx_count_cumul" in cols_lower or num_cols >= 18:
            return 'v3_full'
        return 'engineered_full'

    if num_cols == 18:
        return 'v3_full'
    if num_cols in (13, 14):
        return 'engineered_full'
    if num_cols in (11, 12):
        return 'engineered_legacy'

    return 'raw'


def _prepare_paysim_features(raw_values, num_cols):
    """Prepare PaySim engineered features from any CSV format."""
    csv_format = detect_csv_format(raw_values, num_cols)

    if csv_format == 'v3_full':
        engineered_df = pd.DataFrame([raw_values], columns=PAYSIM_V3_FEATURES)

    elif csv_format == 'engineered_full':
        raw_df = pd.DataFrame([raw_values], columns=ENGINEERED_FULL_COLUMNS)
        if "has_balance_mismatch" in raw_df.columns:
            raw_df = raw_df.drop(columns=["has_balance_mismatch"])
        engineered_df = add_velocity_defaults(raw_df)

    elif csv_format == 'engineered_legacy':
        cols = ENGINEERED_FULL_COLUMNS[:num_cols]
        raw_df = pd.DataFrame([raw_values[:len(cols)]], columns=cols)
        raw_df["upi_type_payment"] = 0
        raw_df["upi_type_transfer"] = 1
        engineered_df = add_velocity_defaults(raw_df)

    else:
        column_names = RAW_PAYSIM_COLUMNS[:num_cols]
        raw_df = pd.DataFrame([raw_values[:len(column_names)]], columns=column_names)
        engineered_df = clean_and_engineer_upi(raw_df)
        engineered_df = add_velocity_defaults(engineered_df)

    return engineered_df


def _prepare_paysim_batch(df: pd.DataFrame, csv_format: str) -> pd.DataFrame:
    """Prepare entire PaySim batch DataFrame for V5 inference."""
    if csv_format == 'v3_full':
        for col in PAYSIM_V3_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
        return df[PAYSIM_V3_FEATURES].copy()

    elif csv_format == 'engineered_full':
        # Drop is_fraud/isfraud if present in synthetic engineered CSVs
        if "is_fraud" in df.columns:
            df = df.drop(columns=["is_fraud"])
        if "isfraud" in df.columns:
            df = df.drop(columns=["isfraud"])
        
        # In synthetic CSVs, names might be 'upi_type_payment' instead of 'upi_type_upi_payment'
        if "upi_type_payment" in df.columns:
            df = df.rename(columns={"upi_type_payment": "upi_type_upi_payment"})
        if "upi_type_transfer" in df.columns:
            df = df.rename(columns={"upi_type_transfer": "upi_type_upi_transfer"})
            
        return add_velocity_defaults_batch(df)

    elif csv_format == 'engineered_legacy':
        df["upi_type_upi_payment"] = 0
        df["upi_type_upi_transfer"] = 1
        return add_velocity_defaults_batch(df)

    else:
        # Raw PaySim
        engineered_df = clean_and_engineer_upi(df)
        return add_velocity_defaults_batch(engineered_df)


# ══════════════════════════════════════════
# LEGACY: Single-row Inference (V3/V4)
# ══════════════════════════════════════════
def run_inference(payload):
    """Legacy entry point for single-transaction V3/V4 inference.
    
    [M1] Noise Guard: Input sanitization applied to all engineered DataFrames
    before they reach the model pipeline.
    """
    raw_values = payload.tabular_features
    txn_type = payload.transaction_type
    model_version = getattr(payload, "model_version", "v3")
    num_cols = len(raw_values)

    if txn_type == "paysim":
        engineered_df = _prepare_paysim_features(raw_values, num_cols)

        # [M1] NOISE GUARD: Sanitize before inference
        engineered_df = noise_guard_sanitize(engineered_df)

        if model_version == "v4":
            v4_models = _load_v4_models()
            if v4_models is None:
                raise ValueError("V4 models not available.")
            from src.v4_ensemble_inference import predict_paysim_v4
            result = predict_paysim_v4(engineered_df, v4_models)
        else:
            result = ensemble_predict(
                transaction_type=txn_type,
                raw_df=engineered_df,
                lstm_sequence=None,
            )

    elif txn_type == "creditcard":
        column_names = CREDITCARD_COLUMNS[:num_cols]
        raw_df = pd.DataFrame([raw_values[:len(column_names)]], columns=column_names)
        engineered_df = raw_df.copy()
        if 'is_fraud' in engineered_df.columns:
            engineered_df = engineered_df.drop(columns=['is_fraud'])
        # [M1] NOISE GUARD: Sanitize credit card input
        engineered_df = noise_guard_sanitize(engineered_df)
        result = ensemble_predict(
            transaction_type=txn_type,
            raw_df=engineered_df,
            lstm_sequence=None,
        )
    else:
        raise ValueError(f"Invalid transaction_type: {txn_type}")

    return result


# ══════════════════════════════════════════
# V5: Batch Inference
# ══════════════════════════════════════════
def run_v5_batch_inference(
    csv_data: list,
    csv_columns: list,
    dataset_type: str,
    production_mode: bool = False,
) -> list:
    """
    V5 batch inference entry point (hardened).

    All 6 robustness mitigations from the stress test audit are active:
      [M1] Noise Guard on input DataFrame
      [M2] Anti-Smurfing detection (in V5 inference)
      [M3] Probability calibration (in V5 inference)
      [M4] Feature health monitoring (in V5 inference)
      [M5] Production mode toggle (leaky feature neutralization)
      [M6] Robust clipping (in velocity defaults + V5 inference)

    Args:
        csv_data: List of row lists
        csv_columns: Column name list
        dataset_type: 'paysim' or 'creditcard'
        production_mode: If True, neutralizes leaky features for
                        real-world deployment. Per Feature_Importance_Report:
                        only 0.2% recall drop without errorbalance* features.

    Returns:
        List of result dicts with decision_label, score, explanation, etc.
    """
    df = pd.DataFrame(csv_data, columns=csv_columns)

    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    if dataset_type == "paysim":
        csv_format = detect_csv_format_by_columns(csv_columns)
        engineered_df = _prepare_paysim_batch(df.copy(), csv_format)

        # [M1] NOISE GUARD: Sanitize entire batch before V5 inference
        engineered_df = noise_guard_sanitize(engineered_df)

        v5 = _load_v5_models()
        if v5 is not None:
            from src.v5_hybrid_inference import predict_paysim_v5_batch
            results = predict_paysim_v5_batch(
                engineered_df, v5["v3"], v5["v4"],
                production_mode=production_mode,
            )
        else:
            # Fallback to V3 row-by-row if V5 models unavailable
            results = []
            for i in range(len(engineered_df)):
                row_df = engineered_df.iloc[[i]].reset_index(drop=True)
                r = ensemble_predict("paysim", row_df, None)
                results.append(r)

    elif dataset_type == "creditcard":
        # Credit card uses V3 ensemble (no V5 specific model)
        results = []
        for i in range(len(df)):
            row_df = df.iloc[[i]].reset_index(drop=True)
            if 'is_fraud' in row_df.columns:
                row_df = row_df.drop(columns=['is_fraud'])
            if 'isfraud' in row_df.columns:
                row_df = row_df.drop(columns=['isfraud'])
            # [M1] NOISE GUARD: Sanitize credit card rows
            row_df = noise_guard_sanitize(row_df)
            r = ensemble_predict("creditcard", row_df, None)
            results.append(r)
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")

    return results
