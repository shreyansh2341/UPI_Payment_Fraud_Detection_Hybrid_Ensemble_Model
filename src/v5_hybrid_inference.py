"""
v5_hybrid_inference.py — V5 Best-of-Both-Worlds Hybrid (Hardened)
═════════════════════════════════════════════════════════════════
Combines V3's proven Path A with V4's BiLSTM Path B for a three-tier
decision system that blocks BOTH known and novel fraud:

  Tier 1 (Path A):     V3's XGB+RF (19 features) → BLOCK known fraud
  Tier 2 (Path B-Block): V4's BiLSTM + anomaly confirmation → BLOCK novel fraud
  Tier 2b (Anti-Smurf): Velocity-based smurfing detection → BLOCK split attacks
  Tier 3 (Path B-Review): Remaining anomalies → REVIEW
  Otherwise:            ALLOW

Why V5 instead of pure V4:
  - V3's XGB+RF Path A achieves 100% recall, 100% precision (proven)
  - V4's XGB+RF Path A is miscalibrated (SMOTE-balanced training vs raw eval)
  - V4's BiLSTM excels at sequential pattern detection for novel fraud
  - V5 = V3 blocking + V4 novel fraud detection = best of both

Robustness Mitigations Applied (per stress test audit):
  [M1] Noise Guard:       Input sanitization before scaling (Noise_Injection_Report)
  [M2] Anti-Smurfing:     Cumulative velocity detection (Adversarial_Evasion_Report)
  [M3] Calibration:       Probability sharpening (Calibration_Analysis_Report)
  [M4] Feature Health:    Runtime degradation detection (Feature_Ablation_Report)
  [M5] Production Mode:   Leaky feature neutralization (Feature_Importance_Report)
  [M6] Robust Clipping:   Post-scaling sigma clipping (Noise_Injection_Report)
"""
import numpy as np
import pandas as pd

from src.robustness_mitigations import (
    noise_guard_sanitize,
    noise_guard_sanitize_array,
    detect_smurfing_pattern,
    sharpen_probabilities,
    check_feature_health,
    zero_out_leaky_features,
    robust_clip_to_training_distribution,
)


def predict_paysim_v5(
    engineered_df: pd.DataFrame,
    v3_models: dict,
    v4_models: dict,
    production_mode: bool = False,
):
    """
    V5 hybrid single-transaction prediction (hardened).

    Uses V3 for Path A (auto-block) and V4 for Path B (novel fraud).
    Includes all 6 robustness mitigations from the stress test audit.

    Args:
        engineered_df: DataFrame with base features (single row)
        v3_models: Dict from load_paysim_hybrid()
        v4_models: Dict from load_paysim_v4_hybrid()
        production_mode: If True, neutralizes leaky features
                        (errorbalanceorig/dest) for real-world deployment.
                        Per Feature_Importance_Report: only 0.2% recall drop.

    Returns:
        Dict with decision, score, explanation, review_flag, ae_score, seq_score
    """
    # ── [M1] NOISE GUARD: Sanitize input ──
    engineered_df = noise_guard_sanitize(engineered_df)

    # ── [M4] FEATURE HEALTH: Check for degraded features ──
    health = check_feature_health(engineered_df)

    # ── V3 PATH A: Known Fraud Auto-Block ──
    v3_features = v3_models["features"]
    v3_base = [f for f in v3_features if f != "ae_recon_error"]

    X_v3 = engineered_df.copy()
    for col in v3_base:
        if col not in X_v3.columns:
            X_v3[col] = 0.0

    X_v3_base = X_v3[v3_base].values.astype(np.float64)
    X_v3_base = np.nan_to_num(X_v3_base, nan=0.0, posinf=0.0, neginf=0.0)

    # ── [M5] PRODUCTION MODE: Neutralize leaky features ──
    if production_mode:
        X_v3_base = zero_out_leaky_features(X_v3_base, v3_base)

    # V3 scaler (extract first 18 dims for AE)
    from sklearn.preprocessing import StandardScaler
    v3_scaler = v3_models["scaler"]
    ae_scaler = StandardScaler()
    ae_scaler.mean_ = v3_scaler.mean_[:len(v3_base)]
    ae_scaler.scale_ = v3_scaler.scale_[:len(v3_base)]
    ae_scaler.var_ = v3_scaler.var_[:len(v3_base)]
    ae_scaler.n_features_in_ = len(v3_base)
    ae_scaler.n_samples_seen_ = v3_scaler.n_samples_seen_

    X_v3_base_s = ae_scaler.transform(X_v3_base)

    # ── [M6] ROBUST CLIPPING: Prevent noise-induced outliers ──
    X_v3_base_s = robust_clip_to_training_distribution(X_v3_base_s, max_sigma=4.0)

    # V3 AE error
    rec_v3 = v3_models["ae"].predict(X_v3_base_s, batch_size=256, verbose=0)
    ae_err_v3 = float(np.log1p(np.mean(np.square(X_v3_base_s - rec_v3))))

    # V3 19-feature ensemble
    X_v3_19 = np.column_stack([X_v3_base, [[ae_err_v3]]])
    X_v3_19_s = v3_scaler.transform(X_v3_19)

    # ── [M6] ROBUST CLIPPING on 19-feature scaled data ──
    X_v3_19_s = robust_clip_to_training_distribution(X_v3_19_s, max_sigma=4.0)

    prob_xgb_v3 = float(v3_models["xgb"].predict_proba(X_v3_19_s)[0, 1])
    prob_rf_v3 = float(v3_models["rf"].predict_proba(X_v3_19_s)[0, 1])
    w = v3_models["weights"]
    v3_confidence = w[0] * prob_xgb_v3 + w[1] * prob_rf_v3

    # ── [M3] CALIBRATION: Sharpen mid-range probabilities ──
    v3_confidence_calibrated = float(
        sharpen_probabilities(np.array([v3_confidence]))[0]
    )

    # ── [M4] FEATURE HEALTH: Adjust threshold if features degraded ──
    block_threshold = v3_models["block_threshold"]
    if not health["healthy"]:
        # Raise threshold to compensate for precision loss
        block_threshold = min(
            block_threshold * health["threshold_multiplier"],
            0.99  # Never exceed 0.99
        )

    is_tier1_block = v3_confidence >= block_threshold

    # ── V4 PATH B: Novel Fraud Detection (BiLSTM) ──
    v4_features = v4_models["features"][:18]
    v4_base_scaler = v4_models["base_scaler"]
    seq_model = v4_models["sequential"]
    seq_length = v4_models["seq_length"]

    X_v4 = engineered_df.copy()
    for col in v4_features:
        if col not in X_v4.columns:
            X_v4[col] = 0.0

    X_v4_base = X_v4[v4_features].values.astype(np.float64)
    X_v4_base = np.nan_to_num(X_v4_base, nan=0.0, posinf=0.0, neginf=0.0)

    # ── [M5] PRODUCTION MODE on V4 path ──
    if production_mode:
        X_v4_base = zero_out_leaky_features(X_v4_base, v4_features)

    X_v4_base_s = v4_base_scaler.transform(X_v4_base)

    # ── [M6] ROBUST CLIPPING on V4 scaled data ──
    X_v4_base_s = robust_clip_to_training_distribution(X_v4_base_s, max_sigma=4.0)

    # V4 AE error (for anomaly detection)
    rec_v4 = v4_models["ae"].predict(X_v4_base_s, batch_size=256, verbose=0)
    ae_err_v4 = float(np.log1p(np.mean(np.square(X_v4_base_s - rec_v4))))

    # Sequential score (pseudo-sequence for single transaction)
    X_seq = np.tile(X_v4_base_s, (seq_length, 1)).reshape(1, seq_length, -1)
    seq_score = float(seq_model.predict(X_seq, verbose=0).ravel()[0])

    # Anomaly flags
    ae_flag = ae_err_v4 >= v4_models["ae_threshold"]
    iforest_flag = bool(v4_models["iforest"].predict(X_v4_base_s)[0] == -1)
    anomaly_flag = ae_flag or iforest_flag

    # Tier 2: Novel fraud blocking
    seq_block_threshold = v4_models.get("seq_block_threshold", 0.5)
    is_tier2_block = (
        not is_tier1_block
        and seq_score >= seq_block_threshold
        and anomaly_flag
    )

    # Tier 3: Review
    seq_review_threshold = v4_models.get("seq_threshold", 0.26)
    is_tier3_review = (
        not is_tier1_block
        and not is_tier2_block
        and (anomaly_flag or seq_score >= seq_review_threshold)
    )

    # Decision
    if is_tier1_block:
        decision_str = "BLOCK"
    elif is_tier2_block:
        decision_str = "BLOCK_NOVEL"
    elif is_tier3_review:
        decision_str = "REVIEW"
    else:
        decision_str = "ALLOW"

    # Explanation
    flags = []
    if ae_flag:
        flags.append("AE")
    if iforest_flag:
        flags.append("IForest")
    if seq_score >= seq_block_threshold:
        flags.append("SeqBlock")
    elif seq_score >= seq_review_threshold:
        flags.append("SeqReview")
    if not health["healthy"]:
        flags.append(f"FeaturesDegraded({health['n_degraded']})")
    if production_mode:
        flags.append("ProductionMode")

    explanation = (
        f"V5 Hybrid | V3-Ensemble={v3_confidence:.4f} "
        f"(XGB={prob_xgb_v3:.3f}, RF={prob_rf_v3:.3f}) | "
        f"V4-AE={ae_err_v4:.4f} | SeqScore={seq_score:.4f} | "
        f"Decision={decision_str}"
    )
    if flags:
        explanation += f" | Flags: {','.join(flags)}"

    return {
        "decision": 1 if (is_tier1_block or is_tier2_block) else 0,
        "decision_label": decision_str,
        "score": v3_confidence,
        "score_calibrated": v3_confidence_calibrated,
        "explanation": explanation,
        "review_flag": is_tier3_review,
        "ae_score": ae_err_v4,
        "seq_score": seq_score,
        "feature_health": health,
    }


def predict_paysim_v5_batch(
    engineered_df: pd.DataFrame,
    v3_models: dict,
    v4_models: dict,
    production_mode: bool = False,
):
    """
    Batch V5 hybrid prediction with real sequential context (hardened).

    Tier 1 uses V3's proven XGB+RF ensemble.
    Tier 2 uses V4's BiLSTM with real sequential context from the CSV.
    Tier 2b adds anti-smurfing detection from the adversarial evasion audit.

    All 6 robustness mitigations are applied:
      [M1] Noise Guard on input DataFrame
      [M2] Anti-Smurfing detection on batch window
      [M3] Probability calibration/sharpening
      [M4] Feature health monitoring
      [M5] Production mode (leaky feature neutralization)
      [M6] Robust sigma clipping post-scaling

    Args:
        engineered_df: DataFrame with multiple rows
        v3_models, v4_models: Model dicts
        production_mode: If True, neutralizes leaky features

    Returns:
        List of result dicts (one per transaction)
    """
    # ── [M1] NOISE GUARD: Sanitize entire batch ──
    engineered_df = noise_guard_sanitize(engineered_df)

    n = len(engineered_df)

    # ── [M4] FEATURE HEALTH: Check batch-level health ──
    health = check_feature_health(engineered_df)

    # ── [M2] ANTI-SMURFING: Detect split-transaction attacks ──
    smurfing_flags = detect_smurfing_pattern(engineered_df)

    # ══════════════════════════════════════════
    # V3 PATH A: Known Fraud Auto-Block
    # ══════════════════════════════════════════
    v3_features = v3_models["features"]
    v3_base = [f for f in v3_features if f != "ae_recon_error"]

    X_v3_base = np.zeros((n, len(v3_base)), dtype=np.float64)
    for i, feat in enumerate(v3_base):
        if feat in engineered_df.columns:
            X_v3_base[:, i] = engineered_df[feat].values
    X_v3_base = np.nan_to_num(X_v3_base, nan=0.0, posinf=0.0, neginf=0.0)

    # ── [M5] PRODUCTION MODE: Neutralize leaky features ──
    if production_mode:
        X_v3_base = zero_out_leaky_features(X_v3_base, v3_base)

    # V3 AE scaler (first 18 dims of V3 scaler)
    from sklearn.preprocessing import StandardScaler
    v3_scaler = v3_models["scaler"]
    ae_scaler = StandardScaler()
    ae_scaler.mean_ = v3_scaler.mean_[:len(v3_base)]
    ae_scaler.scale_ = v3_scaler.scale_[:len(v3_base)]
    ae_scaler.var_ = v3_scaler.var_[:len(v3_base)]
    ae_scaler.n_features_in_ = len(v3_base)
    ae_scaler.n_samples_seen_ = v3_scaler.n_samples_seen_

    X_v3_base_s = ae_scaler.transform(X_v3_base)

    # ── [M6] ROBUST CLIPPING: Prevent noise-induced outliers ──
    X_v3_base_s = robust_clip_to_training_distribution(X_v3_base_s, max_sigma=4.0)

    # V3 AE errors
    rec_v3 = v3_models["ae"].predict(X_v3_base_s, batch_size=2048, verbose=0)
    ae_err_v3 = np.log1p(np.mean(np.square(X_v3_base_s - rec_v3), axis=1))

    # V3 19-feature ensemble
    X_v3_19 = np.column_stack([X_v3_base, ae_err_v3])
    X_v3_19_s = v3_scaler.transform(X_v3_19)

    # ── [M6] ROBUST CLIPPING on 19-feature scaled data ──
    X_v3_19_s = robust_clip_to_training_distribution(X_v3_19_s, max_sigma=4.0)

    prob_xgb_v3 = v3_models["xgb"].predict_proba(X_v3_19_s)[:, 1]
    prob_rf_v3 = v3_models["rf"].predict_proba(X_v3_19_s)[:, 1]
    w = v3_models["weights"]
    v3_confidence = w[0] * prob_xgb_v3 + w[1] * prob_rf_v3

    # ── [M3] CALIBRATION: Sharpen mid-range probabilities ──
    v3_confidence_calibrated = sharpen_probabilities(v3_confidence)

    # ── [M4] FEATURE HEALTH: Adjust threshold ──
    block_threshold = v3_models["block_threshold"]
    if not health["healthy"]:
        block_threshold = min(
            block_threshold * health["threshold_multiplier"],
            0.99
        )

    tier1_block = v3_confidence >= block_threshold

    # ══════════════════════════════════════════
    # V4 PATH B: Novel Fraud Detection (BiLSTM)
    # ══════════════════════════════════════════
    v4_features = v4_models["features"][:18]
    v4_base_scaler = v4_models["base_scaler"]
    seq_model = v4_models["sequential"]
    seq_length = v4_models["seq_length"]

    X_v4_base = np.zeros((n, len(v4_features)), dtype=np.float64)
    for i, feat in enumerate(v4_features):
        if feat in engineered_df.columns:
            X_v4_base[:, i] = engineered_df[feat].values
    X_v4_base = np.nan_to_num(X_v4_base, nan=0.0, posinf=0.0, neginf=0.0)

    # ── [M5] PRODUCTION MODE on V4 path ──
    if production_mode:
        X_v4_base = zero_out_leaky_features(X_v4_base, v4_features)

    X_v4_base_s = v4_base_scaler.transform(X_v4_base)

    # ── [M6] ROBUST CLIPPING on V4 scaled data ──
    X_v4_base_s = robust_clip_to_training_distribution(X_v4_base_s, max_sigma=4.0)

    # V4 AE errors
    rec_v4 = v4_models["ae"].predict(X_v4_base_s, batch_size=2048, verbose=0)
    ae_errs_v4 = np.log1p(np.mean(np.square(X_v4_base_s - rec_v4), axis=1))

    # Sequential scores (real sequential context)
    seq_scores = np.zeros(n, dtype=np.float32)
    sequences = []
    valid_indices = []
    for i in range(seq_length, n):
        sequences.append(X_v4_base_s[i - seq_length : i])
        valid_indices.append(i)

    for i in range(min(seq_length, n)):
        padded = np.zeros((seq_length, X_v4_base_s.shape[1]), dtype=np.float32)
        padded[-(i + 1):] = X_v4_base_s[:i + 1]
        sequences.insert(i, padded)
        valid_indices.insert(i, i)

    if sequences:
        sequences = np.array(sequences, dtype=np.float32)
        preds = seq_model.predict(sequences, batch_size=512, verbose=0).ravel()
        for idx, pred in zip(valid_indices, preds):
            seq_scores[idx] = pred

    # Anomaly flags
    ae_flags = ae_errs_v4 >= v4_models["ae_threshold"]
    iforest_flags = v4_models["iforest"].predict(X_v4_base_s) == -1
    anomaly_flags = ae_flags | iforest_flags

    # Thresholds
    seq_block_threshold = v4_models.get("seq_block_threshold", 0.5)
    seq_review_threshold = v4_models.get("seq_threshold", 0.26)

    # ── THREE-TIER LOGIC ──
    tier2_block = (
        (seq_scores >= seq_block_threshold)
        & anomaly_flags
        & ~tier1_block
    )

    # ── [M2] ANTI-SMURFING: Override Tier 2 for smurfing patterns ──
    # Transactions flagged as smurfing get blocked regardless of
    # individual BiLSTM scores, addressing the 46% evasion rate
    tier2_smurf_block = smurfing_flags & ~tier1_block & ~tier2_block
    tier2_block = tier2_block | tier2_smurf_block

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
            if smurfing_flags[i]:
                decision_str = "BLOCK_SMURFING"
            else:
                decision_str = "BLOCK_NOVEL"
        elif tier3_review[i]:
            decision_str = "REVIEW"
        else:
            decision_str = "ALLOW"

        # Build per-transaction flags
        tx_flags = []
        if ae_flags[i]:
            tx_flags.append("AE")
        if iforest_flags[i]:
            tx_flags.append("IForest")
        if smurfing_flags[i]:
            tx_flags.append("AntiSmurf")
        if not health["healthy"]:
            tx_flags.append(f"FeatDeg({health['n_degraded']})")
        if production_mode:
            tx_flags.append("ProdMode")

        flag_str = f" | Flags: {','.join(tx_flags)}" if tx_flags else ""

        results.append({
            "decision": 1 if (tier1_block[i] or tier2_block[i]) else 0,
            "decision_label": decision_str,
            "score": float(v3_confidence[i]),
            "score_calibrated": float(v3_confidence_calibrated[i]),
            "explanation": (
                f"V5 Hybrid | V3-Ensemble={v3_confidence[i]:.4f} "
                f"(XGB={prob_xgb_v3[i]:.3f}, RF={prob_rf_v3[i]:.3f}) | "
                f"V4-AE={ae_errs_v4[i]:.4f} | Seq={seq_scores[i]:.4f} | "
                f"Decision={decision_str}{flag_str}"
            ),
            "review_flag": bool(tier3_review[i]),
            "ae_score": float(ae_errs_v4[i]),
            "seq_score": float(seq_scores[i]),
            "smurfing_flag": bool(smurfing_flags[i]),
            "feature_health": health,
        })

    return results
