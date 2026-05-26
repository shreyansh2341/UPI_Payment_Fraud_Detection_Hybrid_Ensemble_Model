"""
final_ensemble_inference.py — V3 Hybrid
────────────────────────────────────────
Two-path fraud detection:
  Path A (Auto-Block): XGB+RF with 19 features (18 + ae_recon_error)
  Path B (Flag for Review): Standalone AE + IForest anomaly detection
"""
import numpy as np
import pandas as pd
from src.model_loader import load_paysim_hybrid, load_creditcard_models

# Lazy model caches (loaded on first use instead of import time)
_PAYSIM = None
_CREDITCARD = None


def _get_paysim():
    global _PAYSIM
    if _PAYSIM is None:
        _PAYSIM = load_paysim_hybrid()
    return _PAYSIM


def _get_creditcard():
    global _CREDITCARD
    if _CREDITCARD is None:
        _CREDITCARD = load_creditcard_models()
    return _CREDITCARD


def predict_paysim(engineered_df: pd.DataFrame, lstm_sequence=None):
    """
    PaySim hybrid prediction.
    Returns: {decision: str, score: float, explanation: str,
              review_flag: bool, ae_score: float}
    """
    PAYSIM = _get_paysim()
    base_18 = PAYSIM["features"][:18]
    scaler_19 = PAYSIM["scaler"]
    ae = PAYSIM["ae"]

    X = engineered_df.copy()
    for col in base_18:
        if col not in X.columns:
            X[col] = 0.0

    X_18 = X[base_18].values.astype(np.float64)
    X_18 = np.nan_to_num(X_18, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale 18 features for AE (using first 18 dims of 19-feature scaler)
    mean_18 = scaler_19.mean_[:18]
    scale_18 = scaler_19.scale_[:18]
    X_18_s = (X_18 - mean_18) / scale_18

    # Compute AE reconstruction error
    rec = ae.predict(X_18_s, batch_size=256, verbose=0)
    ae_err = float(np.log1p(np.mean(np.square(X_18_s - rec))))

    # ── PATH A: Auto-Block (XGB+RF with 19 features) ──
    X_19 = np.column_stack([X_18, [[ae_err]]])
    X_19_s = scaler_19.transform(X_19)

    prob_xgb = float(PAYSIM["xgb"].predict_proba(X_19_s)[0, 1])
    prob_rf = float(PAYSIM["rf"].predict_proba(X_19_s)[0, 1])
    confidence = 0.5 * prob_xgb + 0.5 * prob_rf
    auto_block = confidence >= PAYSIM["block_threshold"]

    # ── PATH B: Flag for Review (AE + IForest) ──
    ae_flag = ae_err >= PAYSIM["ae_threshold"]
    iforest_flag = bool(PAYSIM["iforest"].predict(X_18_s)[0] == -1)
    review_flag = (ae_flag or iforest_flag) and not auto_block

    # Decision
    if auto_block:
        decision_str = "BLOCK"
    elif review_flag:
        decision_str = "REVIEW"
    else:
        decision_str = "ALLOW"

    explanation = (
        f"Ensemble={confidence:.4f} (XGB={prob_xgb:.3f}, RF={prob_rf:.3f}) | "
        f"AE={ae_err:.4f} | Decision={decision_str}"
    )

    return {
        "decision": 1 if auto_block else 0,
        "decision_label": decision_str,
        "score": confidence,
        "explanation": explanation,
        "review_flag": review_flag,
        "ae_score": ae_err,
    }


def predict_creditcard(engineered_df: pd.DataFrame):
    """Credit Card ensemble prediction (XGBoost + Random Forest)."""
    CREDITCARD = _get_creditcard()
    features = CREDITCARD["features"]

    missing = set(features) - set(engineered_df.columns)
    if missing:
        raise ValueError(f"Missing Credit Card features: {missing}")

    X = engineered_df[features]
    X_scaled = CREDITCARD["scaler"].transform(X)

    prob_xgb = CREDITCARD["xgb"].predict_proba(X_scaled)[0, 1]
    prob_rf = CREDITCARD["rf"].predict_proba(X_scaled)[0, 1]

    w_xgb, w_rf = CREDITCARD["weights"]
    prob = w_xgb * prob_xgb + w_rf * prob_rf
    decision = prob >= CREDITCARD["threshold"]

    explanation = (
        f"Ensemble={prob:.4f} (XGB={prob_xgb:.3f}, RF={prob_rf:.3f}, "
        f"weights={w_xgb:.1f}/{w_rf:.1f})"
    )

    return {
        "decision": bool(decision),
        "decision_label": "FRAUD" if decision else "LEGIT",
        "score": float(prob),
        "explanation": explanation,
        "review_flag": False,
        "ae_score": 0.0,
    }


def ensemble_predict(transaction_type, raw_df, lstm_sequence=None):
    """Unified entry point."""
    if transaction_type == "paysim":
        return predict_paysim(raw_df, lstm_sequence)

    if transaction_type == "creditcard":
        return predict_creditcard(raw_df)

    raise ValueError("Invalid transaction_type")