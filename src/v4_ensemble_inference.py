"""
v4_ensemble_inference.py — V4 Three-Tier Hybrid
═════════════════════════════════════════════════
Enhanced fraud detection with three decision tiers:

  Tier 1 (Path A):     Known fraud auto-block via XGB+RF ensemble (20 features)
  Tier 2 (Path B-Block): Novel fraud block via BiLSTM + dual anomaly confirmation
  Tier 3 (Path B-Review): Uncertain anomalies flagged for human review
  Otherwise:            ALLOW

Decision labels: "BLOCK", "BLOCK_NOVEL", "REVIEW", "ALLOW"
"""
import numpy as np
import pandas as pd


def predict_paysim_v4(engineered_df: pd.DataFrame, v4_models: dict):
    """
    V4 three-tier single-transaction prediction.

    Pipeline:
      1. Extract 18 base features → scale → AE reconstruction error
      2. Build pseudo-sequence → compute BiLSTM sequential score
      3. Stack 20 features (18 + ae_error + seq_score) → scale → XGB+RF ensemble
      4. Tier 1: ensemble confidence >= path_a_threshold → BLOCK
      5. Tier 2: seq_score >= seq_block_threshold AND anomaly flag → BLOCK_NOVEL
      6. Tier 3: anomaly flag OR seq_score >= review_threshold → REVIEW
      7. None of the above → ALLOW

    Args:
        engineered_df: DataFrame with 18 base features (single row)
        v4_models: Dict from load_paysim_v4_hybrid()

    Returns:
        Dict with decision, score, explanation, review_flag, ae_score, seq_score
    """
    base_features = v4_models["features"][:18]
    base_scaler = v4_models["base_scaler"]
    scaler_20 = v4_models["scaler"]
    ae = v4_models["ae"]
    seq_model = v4_models["sequential"]
    seq_length = v4_models["seq_length"]

    # Prepare 18 base features
    X = engineered_df.copy()
    for col in base_features:
        if col not in X.columns:
            X[col] = 0.0

    X_18 = X[base_features].values.astype(np.float64)
    X_18 = np.nan_to_num(X_18, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale and compute AE reconstruction error
    X_18_s = base_scaler.transform(X_18)
    rec = ae.predict(X_18_s, batch_size=256, verbose=0)
    ae_err = float(np.log1p(np.mean(np.square(X_18_s - rec))))

    # Compute sequential score (pseudo-sequence for single transaction)
    X_seq = np.tile(X_18_s, (seq_length, 1)).reshape(1, seq_length, -1)
    seq_score = float(seq_model.predict(X_seq, verbose=0).ravel()[0])

    # Build 20-feature vector → XGB+RF ensemble
    X_20 = np.column_stack([X_18, [[ae_err]], [[seq_score]]])
    X_20_s = scaler_20.transform(X_20)

    prob_xgb = float(v4_models["xgb"].predict_proba(X_20_s)[0, 1])
    prob_rf = float(v4_models["rf"].predict_proba(X_20_s)[0, 1])
    w_xgb, w_rf = v4_models["weights"]
    confidence = w_xgb * prob_xgb + w_rf * prob_rf

    # Anomaly flags
    ae_flag = ae_err >= v4_models["ae_threshold"]
    iforest_flag = bool(v4_models["iforest"].predict(X_18_s)[0] == -1)
    anomaly_flag = ae_flag or iforest_flag

    # ── THREE-TIER DECISION LOGIC ──

    # Tier 1: Path A — known fraud auto-block
    is_tier1_block = confidence >= v4_models["block_threshold"]

    # Tier 2: Path B-Block — novel fraud blocking
    seq_block_threshold = v4_models.get("seq_block_threshold", 0.5)
    is_tier2_block = (
        not is_tier1_block
        and seq_score >= seq_block_threshold
        and anomaly_flag
    )

    # Tier 3: Path B-Review — uncertain anomalies
    seq_review_threshold = v4_models.get("seq_threshold", 0.26)
    is_tier3_review = (
        not is_tier1_block
        and not is_tier2_block
        and (anomaly_flag or seq_score >= seq_review_threshold)
    )

    # Decision label
    if is_tier1_block:
        decision_str = "BLOCK"
    elif is_tier2_block:
        decision_str = "BLOCK_NOVEL"
    elif is_tier3_review:
        decision_str = "REVIEW"
    else:
        decision_str = "ALLOW"

    # Build explanation
    flags = []
    if ae_flag:
        flags.append("AE")
    if iforest_flag:
        flags.append("IForest")
    if seq_score >= seq_block_threshold:
        flags.append("SeqBlock")
    elif seq_score >= seq_review_threshold:
        flags.append("SeqReview")

    explanation = (
        f"V4 Ensemble={confidence:.4f} (XGB={prob_xgb:.3f}, RF={prob_rf:.3f}) | "
        f"AE={ae_err:.4f} | SeqScore={seq_score:.4f} | "
        f"Decision={decision_str}"
    )
    if flags:
        explanation += f" | Flags: {','.join(flags)}"

    return {
        "decision": 1 if (is_tier1_block or is_tier2_block) else 0,
        "decision_label": decision_str,
        "score": confidence,
        "explanation": explanation,
        "review_flag": is_tier3_review,
        "ae_score": ae_err,
        "seq_score": seq_score,
    }


def predict_paysim_v4_batch(engineered_df: pd.DataFrame, v4_models: dict):
    """
    Batch V4 three-tier prediction with real sequential context.

    Unlike single-transaction mode, batch mode builds REAL sequences
    from the transaction history in the CSV for accurate BiLSTM scoring.

    Args:
        engineered_df: DataFrame with multiple rows, 18 base features each
        v4_models: Dict from load_paysim_v4_hybrid()

    Returns:
        List of result dicts (one per transaction)
    """
    base_features = v4_models["features"][:18]
    base_scaler = v4_models["base_scaler"]
    scaler_20 = v4_models["scaler"]
    ae = v4_models["ae"]
    seq_model = v4_models["sequential"]
    seq_length = v4_models["seq_length"]

    # Prepare all features
    X = engineered_df.copy()
    for col in base_features:
        if col not in X.columns:
            X[col] = 0.0

    X_18 = X[base_features].values.astype(np.float64)
    X_18 = np.nan_to_num(X_18, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale
    X_18_s = base_scaler.transform(X_18)

    # AE errors (batch)
    rec = ae.predict(X_18_s, batch_size=2048, verbose=0)
    ae_errs = np.log1p(np.mean(np.square(X_18_s - rec), axis=1))

    # Sequential scores (batch — real sequential context)
    n = len(X_18_s)
    seq_scores = np.zeros(n, dtype=np.float32)

    sequences = []
    valid_indices = []
    for i in range(seq_length, n):
        sequences.append(X_18_s[i - seq_length : i])
        valid_indices.append(i)

    # Pad early transactions (< seq_length) with zeros
    for i in range(min(seq_length, n)):
        padded = np.zeros((seq_length, X_18_s.shape[1]), dtype=np.float32)
        padded[-(i + 1):] = X_18_s[:i + 1]
        sequences.insert(i, padded)
        valid_indices.insert(i, i)

    if sequences:
        sequences = np.array(sequences, dtype=np.float32)
        preds = seq_model.predict(sequences, batch_size=512, verbose=0).ravel()
        for idx, pred in zip(valid_indices, preds):
            seq_scores[idx] = pred

    # Build 20-feature matrix
    X_20 = np.column_stack([X_18, ae_errs, seq_scores])
    X_20_s = scaler_20.transform(X_20)

    # Path A: XGB+RF ensemble
    prob_xgb = v4_models["xgb"].predict_proba(X_20_s)[:, 1]
    prob_rf = v4_models["rf"].predict_proba(X_20_s)[:, 1]
    w_xgb, w_rf = v4_models["weights"]
    confidence = w_xgb * prob_xgb + w_rf * prob_rf

    # Anomaly flags
    ae_flags = ae_errs >= v4_models["ae_threshold"]
    iforest_flags = v4_models["iforest"].predict(X_18_s) == -1
    anomaly_flags = ae_flags | iforest_flags

    # Thresholds
    path_a_threshold = v4_models["block_threshold"]
    seq_block_threshold = v4_models.get("seq_block_threshold", 0.5)
    seq_review_threshold = v4_models.get("seq_threshold", 0.26)

    # ── THREE-TIER DECISION LOGIC (VECTORIZED) ──

    # Tier 1: Path A auto-block
    tier1_block = confidence >= path_a_threshold

    # Tier 2: Path B-Block (novel fraud)
    tier2_block = (
        (seq_scores >= seq_block_threshold)
        & anomaly_flags
        & ~tier1_block
    )

    # Tier 3: Path B-Review (uncertain)
    tier3_review = (
        (anomaly_flags | (seq_scores >= seq_review_threshold))
        & ~tier1_block
        & ~tier2_block
    )

    # Build results
    results = []
    for i in range(n):
        if tier1_block[i]:
            decision_str = "BLOCK"
        elif tier2_block[i]:
            decision_str = "BLOCK_NOVEL"
        elif tier3_review[i]:
            decision_str = "REVIEW"
        else:
            decision_str = "ALLOW"

        results.append({
            "decision": 1 if (tier1_block[i] or tier2_block[i]) else 0,
            "decision_label": decision_str,
            "score": float(confidence[i]),
            "explanation": (
                f"V4 Ensemble={confidence[i]:.4f} "
                f"(XGB={prob_xgb[i]:.3f}, RF={prob_rf[i]:.3f}) | "
                f"AE={ae_errs[i]:.4f} | Seq={seq_scores[i]:.4f} | "
                f"Decision={decision_str}"
            ),
            "review_flag": bool(tier3_review[i]),
            "ae_score": float(ae_errs[i]),
            "seq_score": float(seq_scores[i]),
        })

    return results
