"""
retune_v4_threshold.py — Three-Tier Threshold Optimization
═══════════════════════════════════════════════════════════
Fixes V4's miscalibrated Path A threshold and adds Path B-Block
thresholds for a three-tier decision system:

  Tier 1 (Path A):     Known fraud auto-block via XGB+RF ensemble
  Tier 2 (Path B-Block): Novel fraud block via high-confidence BiLSTM + anomaly
  Tier 3 (Path B-Review): Uncertain anomalies flagged for human review

This script loads the EXISTING V4 models (no retraining needed) and
recomputes thresholds using the validation set with raw (non-SMOTE) data,
exactly matching how evaluation works in production.

Usage:
  python experiments/paysim/threshold_tuning/retune_v4_threshold.py
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Keras cross-version compatibility patch
_original_dense_init = tf.keras.layers.Dense.__init__

def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    _original_dense_init(self, *args, **kwargs)

tf.keras.layers.Dense.__init__ = _patched_dense_init

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    f1_score,
)

# ══════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════
print("=" * 70)
print(" V4 THREE-TIER THRESHOLD OPTIMIZATION")
print("=" * 70)

DATA_PATH = "data/cleaned_paysim_lstm.csv"
V4_DIR = "models/paysim_v4_experiment"

print(f"\n📂 Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower().str.strip()
df = df.sort_values("step").reset_index(drop=True)

# Engineer velocity features (same as training)
print("   Engineering velocity features...")
if "hour" not in df.columns:
    df["hour"] = df["step"] % 24
if "dayofweek" not in df.columns:
    df["dayofweek"] = (df["step"] // 24) % 7
if "is_weekend" not in df.columns:
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)

for col in ["upi_type_upi_payment", "upi_type_upi_transfer"]:
    if col in df.columns:
        df[col] = df[col].astype(np.int8)

df["tx_count_cumul"] = df.groupby("nameorig").cumcount() + 1
df["amount_cumul"] = df.groupby("nameorig")["amount"].cumsum()
df["amt_vs_avg"] = df["amount"] / (df["amount_cumul"] / df["tx_count_cumul"] + 1e-6)
df["time_since_last"] = df.groupby("nameorig")["step"].diff().fillna(48).clip(0, 96)
df["amt_to_bal_ratio"] = df["amount"] / (df["oldbalanceorg"] + 1e-6)
df["balance_velocity"] = (
    (df["newbalanceorig"] - df["oldbalanceorg"]) / (df["amount"] + 1e-6)
)

df["amount_cumul"] = np.log1p(df["amount_cumul"].clip(0))
df["tx_count_cumul"] = np.log1p(df["tx_count_cumul"])
df["amt_to_bal_ratio"] = np.log1p(df["amt_to_bal_ratio"].clip(0))

TARGET = "isfraud"

# Time-aware split (same as training: 70/15/15)
n = len(df)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

df_val = df.iloc[train_end:val_end].reset_index(drop=True)
df_test = df.iloc[val_end:].reset_index(drop=True)

y_val = df_val[TARGET].values.astype(np.int32)
y_test = df_test[TARGET].values.astype(np.int32)

print(f"   Val set:  {len(y_val):,} transactions ({y_val.sum():,} frauds, {y_val.mean():.4%})")
print(f"   Test set: {len(y_test):,} transactions ({y_test.sum():,} frauds, {y_test.mean():.4%})")


# ══════════════════════════════════════════════════════
# LOAD V4 MODELS
# ══════════════════════════════════════════════════════
print(f"\n📦 Loading V4 models from {V4_DIR}...")
from src.v4_layers import BahdanauAttention

v4_models = {
    "xgb": joblib.load(f"{V4_DIR}/paysim_v4_xgb.pkl"),
    "rf": joblib.load(f"{V4_DIR}/paysim_v4_rf.pkl"),
    "ae": tf.keras.models.load_model(
        f"{V4_DIR}/paysim_v4_ae.keras", compile=False, safe_mode=False
    ),
    "sequential": tf.keras.models.load_model(
        f"{V4_DIR}/paysim_v4_sequential_winner.keras",
        compile=False, safe_mode=False,
        custom_objects={"BahdanauAttention": BahdanauAttention},
    ),
    "iforest": joblib.load(f"{V4_DIR}/paysim_v4_iforest.pkl"),
    "base_scaler": joblib.load(f"{V4_DIR}/paysim_v4_base_scaler.pkl"),
    "scaler_20": joblib.load(f"{V4_DIR}/paysim_v4_scaler.pkl"),
    "features": joblib.load(f"{V4_DIR}/paysim_v4_features.pkl"),
    "seq_length": joblib.load(f"{V4_DIR}/paysim_v4_seq_length.pkl"),
}
print("   V4 loaded ✅")


# ══════════════════════════════════════════════════════
# COMPUTE PREDICTIONS (Raw pipeline, no SMOTE)
# ══════════════════════════════════════════════════════
def compute_pipeline(df_split, v4_models, label="split"):
    """
    Run the full V4 inference pipeline on a data split.
    Returns all intermediate signals needed for threshold tuning.
    """
    features = v4_models["features"]
    base_scaler = v4_models["base_scaler"]
    scaler_20 = v4_models["scaler_20"]
    ae = v4_models["ae"]
    seq_model = v4_models["sequential"]
    seq_length = v4_models["seq_length"]

    # Extract base features
    X_base = np.zeros((len(df_split), len(features)), dtype=np.float64)
    for i, feat in enumerate(features):
        if feat in df_split.columns:
            X_base[:, i] = df_split[feat].values
    X_base = np.nan_to_num(X_base, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale base features
    X_base_scaled = base_scaler.transform(X_base)

    # AE reconstruction error
    rec = ae.predict(X_base_scaled, batch_size=2048, verbose=0)
    ae_errors = np.log1p(np.mean(np.square(X_base_scaled - rec), axis=1))

    # Sequential scores (using real sequential context from data)
    n_samples = len(X_base_scaled)
    seq_scores = np.zeros(n_samples, dtype=np.float32)
    sequences = []
    valid_indices = []
    for i in range(seq_length, n_samples):
        sequences.append(X_base_scaled[i - seq_length : i])
        valid_indices.append(i)
    if sequences:
        sequences = np.array(sequences, dtype=np.float32)
        preds = seq_model.predict(sequences, batch_size=512, verbose=0).ravel()
        for idx, pred in zip(valid_indices, preds):
            seq_scores[idx] = pred

    # 20-feature ensemble
    X_20 = np.column_stack([X_base, ae_errors, seq_scores])
    X_20_scaled = scaler_20.transform(X_20)

    # XGB + RF probabilities
    prob_xgb = v4_models["xgb"].predict_proba(X_20_scaled)[:, 1]
    prob_rf = v4_models["rf"].predict_proba(X_20_scaled)[:, 1]
    ensemble_confidence = 0.5 * prob_xgb + 0.5 * prob_rf

    # Anomaly flags
    ae_flags = ae_errors >= float(np.load(f"{V4_DIR}/paysim_v4_ae_threshold.npy")[0])
    iforest_flags = v4_models["iforest"].predict(X_base_scaled) == -1
    anomaly_flags = ae_flags | iforest_flags

    print(f"   {label}: computed {n_samples:,} predictions")
    print(f"     Ensemble confidence — min: {ensemble_confidence.min():.4f}, "
          f"max: {ensemble_confidence.max():.4f}, mean: {ensemble_confidence.mean():.4f}")
    print(f"     Seq scores — min: {seq_scores.min():.4f}, "
          f"max: {seq_scores.max():.4f}, mean: {seq_scores.mean():.4f}")
    print(f"     AE flags: {ae_flags.sum():,}, IForest flags: {iforest_flags.sum():,}")

    return {
        "ensemble_confidence": ensemble_confidence,
        "seq_scores": seq_scores,
        "ae_errors": ae_errors,
        "ae_flags": ae_flags,
        "iforest_flags": iforest_flags,
        "anomaly_flags": anomaly_flags,
    }


print("\n🔮 Computing validation set predictions...")
val_signals = compute_pipeline(df_val, v4_models, "Validation")

print("\n🔮 Computing test set predictions...")
test_signals = compute_pipeline(df_test, v4_models, "Test")


# ══════════════════════════════════════════════════════
# TIER 1: PATH A THRESHOLD (Known Fraud Auto-Block)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(" TIER 1: PATH A — KNOWN FRAUD AUTO-BLOCK THRESHOLD")
print("=" * 70)

# Sweep thresholds on validation set
precision_arr, recall_arr, thresholds_arr = precision_recall_curve(
    y_val, val_signals["ensemble_confidence"]
)
precision_arr = precision_arr[:-1]
recall_arr = recall_arr[:-1]

# Target: highest precision where recall >= 80%
TARGET_RECALL = 0.80
valid_idx = np.where(recall_arr >= TARGET_RECALL)[0]

if len(valid_idx) > 0:
    best_idx = valid_idx[-1]  # Highest threshold meeting recall target
    path_a_threshold = float(thresholds_arr[best_idx])
    path_a_recall = recall_arr[best_idx]
    path_a_precision = precision_arr[best_idx]
else:
    # Fallback: best F1
    f1_arr = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-12)
    best_idx = np.argmax(f1_arr)
    path_a_threshold = float(thresholds_arr[best_idx])
    path_a_recall = recall_arr[best_idx]
    path_a_precision = precision_arr[best_idx]
    print("   ⚠️  Could not meet 80% recall target, using best F1 threshold")

print(f"\n   Path A threshold: {path_a_threshold:.6f}")
print(f"   Val recall:       {path_a_recall:.4f}")
print(f"   Val precision:    {path_a_precision:.4f}")

# Also show a sweep table for transparency
print(f"\n   {'Threshold':>12} {'Recall':>10} {'Precision':>12} {'F1':>8} {'TP':>8} {'FP':>8}")
print(f"   {'─' * 62}")
for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    preds = (val_signals["ensemble_confidence"] >= t).astype(int)
    tp = ((preds == 1) & (y_val == 1)).sum()
    fp = ((preds == 1) & (y_val == 0)).sum()
    fn = ((preds == 0) & (y_val == 1)).sum()
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    print(f"   {t:>12.2f} {rec:>10.4f} {prec:>12.4f} {f1:>8.4f} {tp:>8,} {fp:>8,}")


# ══════════════════════════════════════════════════════
# TIER 2: PATH B-BLOCK THRESHOLD (Novel Fraud Blocking)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(" TIER 2: PATH B-BLOCK — NOVEL FRAUD BLOCKING THRESHOLD")
print("=" * 70)

"""
Strategy: Find a sequential score threshold that blocks novel fraud
with high confidence. "Novel fraud" = fraud NOT caught by Path A.

The blocking condition is:
  seq_score >= seq_block_threshold AND (ae_flag OR iforest_flag)

The dual-confirmation requirement reduces false positives — we only
block when BOTH the sequential model AND an anomaly detector agree.
"""

# Identify novel frauds: frauds that Path A misses on validation set
path_a_blocked_val = val_signals["ensemble_confidence"] >= path_a_threshold
novel_fraud_mask_val = (y_val == 1) & ~path_a_blocked_val
legit_mask_val = y_val == 0

n_novel_frauds = novel_fraud_mask_val.sum()
print(f"\n   Novel frauds (missed by Path A): {n_novel_frauds:,}")

if n_novel_frauds > 0:
    # Sequential scores for novel frauds vs legitimate
    novel_fraud_seq_scores = val_signals["seq_scores"][novel_fraud_mask_val]
    legit_seq_scores = val_signals["seq_scores"][legit_mask_val]

    print(f"   Novel fraud seq scores — mean: {novel_fraud_seq_scores.mean():.4f}, "
          f"median: {np.median(novel_fraud_seq_scores):.4f}, max: {novel_fraud_seq_scores.max():.4f}")
    print(f"   Legit seq scores      — mean: {legit_seq_scores.mean():.4f}, "
          f"median: {np.median(legit_seq_scores):.4f}, 99th: {np.percentile(legit_seq_scores, 99):.4f}")

    # Sweep seq_block_threshold to find optimal novel fraud blocking
    print(f"\n   {'SeqThresh':>12} {'NovelBlock':>12} {'NovelRate':>12} {'FP':>8} {'FPR':>10}")
    print(f"   {'─' * 56}")

    best_seq_block_threshold = 0.5
    best_novel_catch = 0
    best_fpr = 1.0

    sweep_thresholds = np.arange(0.05, 0.95, 0.05)
    for t in sweep_thresholds:
        # Tier 2 condition: seq_score >= t AND anomaly flag
        tier2_block = (val_signals["seq_scores"] >= t) & val_signals["anomaly_flags"]

        # How many novel frauds does Tier 2 block?
        novel_blocked = (tier2_block & novel_fraud_mask_val).sum()
        novel_rate = novel_blocked / n_novel_frauds if n_novel_frauds > 0 else 0

        # False positives: legitimate transactions blocked by Tier 2
        fp_tier2 = (tier2_block & legit_mask_val).sum()
        fpr_tier2 = fp_tier2 / legit_mask_val.sum() if legit_mask_val.sum() > 0 else 0

        print(f"   {t:>12.2f} {novel_blocked:>12,} {novel_rate:>12.4f} {fp_tier2:>8,} {fpr_tier2:>10.6f}")

        # Select: maximize novel catch rate while FPR < 1%
        if novel_rate > best_novel_catch and fpr_tier2 < 0.01:
            best_novel_catch = novel_rate
            best_seq_block_threshold = float(t)
            best_fpr = fpr_tier2
        elif novel_rate == best_novel_catch and fpr_tier2 < best_fpr:
            best_seq_block_threshold = float(t)
            best_fpr = fpr_tier2

    seq_block_threshold = best_seq_block_threshold
    print(f"\n   ✅ Selected seq_block_threshold: {seq_block_threshold:.4f}")
    print(f"      Novel fraud catch rate: {best_novel_catch:.4f}")
    print(f"      False positive rate:    {best_fpr:.6f}")
else:
    # Path A catches everything — no novel fraud to tune for
    seq_block_threshold = 0.5
    print("   Path A catches all fraud — setting default seq_block_threshold: 0.5")


# ══════════════════════════════════════════════════════
# TIER 3: PATH B-REVIEW THRESHOLD (Uncertain Cases)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(" TIER 3: PATH B-REVIEW — SEQUENTIAL REVIEW THRESHOLD")
print("=" * 70)

# 95th percentile of legitimate sequential scores
seq_review_threshold = float(np.percentile(
    val_signals["seq_scores"][legit_mask_val], 95
))
print(f"   seq_review_threshold (95th pct legit): {seq_review_threshold:.6f}")


# ══════════════════════════════════════════════════════
# SAVE ALL THRESHOLDS
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(" SAVING THRESHOLDS")
print("=" * 70)

np.save(f"{V4_DIR}/paysim_v4_threshold.npy", np.array([path_a_threshold]))
np.save(f"{V4_DIR}/paysim_v4_seq_block_threshold.npy", np.array([seq_block_threshold]))
np.save(f"{V4_DIR}/paysim_v4_seq_threshold.npy", np.array([seq_review_threshold]))

print(f"   Path A threshold:      {path_a_threshold:.6f} → paysim_v4_threshold.npy")
print(f"   Path B-Block threshold: {seq_block_threshold:.6f} → paysim_v4_seq_block_threshold.npy")
print(f"   Path B-Review threshold: {seq_review_threshold:.6f} → paysim_v4_seq_threshold.npy")


# ══════════════════════════════════════════════════════
# EVALUATION ON TEST SET (Three-Tier System)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(" THREE-TIER EVALUATION ON TEST SET")
print("=" * 70)

# Tier 1: Path A auto-block
tier1_block = test_signals["ensemble_confidence"] >= path_a_threshold

# Tier 2: Path B-Block (novel fraud)
# Condition: seq_score >= seq_block_threshold AND anomaly flag AND NOT already blocked
tier2_block = (
    (test_signals["seq_scores"] >= seq_block_threshold)
    & test_signals["anomaly_flags"]
    & ~tier1_block
)

# Tier 3: Path B-Review (uncertain)
# Condition: anomaly flag OR seq_score >= review threshold, AND NOT blocked by Tier 1/2
tier3_review = (
    (test_signals["anomaly_flags"]
     | (test_signals["seq_scores"] >= seq_review_threshold))
    & ~tier1_block
    & ~tier2_block
)

# Combined block = Tier 1 + Tier 2
total_block = tier1_block | tier2_block

# Compute metrics
def compute_tier_metrics(y_true, tier_predictions, tier_name):
    tp = ((tier_predictions) & (y_true == 1)).sum()
    fp = ((tier_predictions) & (y_true == 0)).sum()
    fn = ((~tier_predictions) & (y_true == 1)).sum()
    tn = ((~tier_predictions) & (y_true == 0)).sum()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "recall": recall, "precision": precision, "f1": f1, "fpr": fpr,
    }


tier1_metrics = compute_tier_metrics(y_test, tier1_block, "Path A")
tier2_metrics = compute_tier_metrics(y_test, tier2_block, "Path B-Block")
total_metrics = compute_tier_metrics(y_test, total_block, "Total Block")

# Novel fraud analysis
novel_fraud_test = (y_test == 1) & ~tier1_block
novel_caught_by_tier2 = (tier2_block & (y_test == 1)).sum()
novel_caught_by_review = (tier3_review & novel_fraud_test & ~tier2_block).sum()
total_fraud = y_test.sum()

print(f"\n   TIER 1 — Path A (Known Fraud Auto-Block)")
print(f"   ─────────────────────────────────────────")
print(f"   Threshold:  {path_a_threshold:.6f}")
print(f"   Blocked:    {tier1_block.sum():,} transactions")
print(f"   TP={tier1_metrics['tp']:,}  FP={tier1_metrics['fp']:,}  "
      f"FN={tier1_metrics['fn']:,}  TN={tier1_metrics['tn']:,}")
print(f"   Recall: {tier1_metrics['recall']:.4f}  "
      f"Precision: {tier1_metrics['precision']:.4f}  "
      f"F1: {tier1_metrics['f1']:.4f}")

print(f"\n   TIER 2 — Path B-Block (Novel Fraud Blocking)")
print(f"   ──────────────────────────────────────────────")
print(f"   Seq threshold: {seq_block_threshold:.6f}")
print(f"   Blocked:       {tier2_block.sum():,} transactions")
print(f"   Fraud caught:  {novel_caught_by_tier2:,} novel frauds")
print(f"   FP:            {tier2_metrics['fp']:,}")

print(f"\n   TIER 3 — Path B-Review (Uncertain)")
print(f"   ────────────────────────────────────")
print(f"   Reviews:       {tier3_review.sum():,} transactions")
print(f"   Fraud caught:  {novel_caught_by_review:,} additional frauds in review")

print(f"\n   COMBINED RESULTS")
print(f"   ═══════════════════════════════════════════")
print(f"   Total fraud:            {total_fraud:,}")
print(f"   Tier 1 blocked:         {tier1_metrics['tp']:,} ({tier1_metrics['tp']/total_fraud:.1%})")
print(f"   Tier 2 blocked:         {novel_caught_by_tier2:,} ({novel_caught_by_tier2/total_fraud:.1%})")
print(f"   Total blocked:          {total_metrics['tp']:,} ({total_metrics['tp']/total_fraud:.1%})")
print(f"   Sent to review:         {novel_caught_by_review:,} ({novel_caught_by_review/total_fraud:.1%})")
print(f"   Escaped (missed):       {total_fraud - total_metrics['tp'] - novel_caught_by_review:,}")
print(f"   Combined recall (block):{total_metrics['recall']:.4f}")
print(f"   Combined precision:     {total_metrics['precision']:.4f}")
print(f"   Combined F1:            {total_metrics['f1']:.4f}")
print(f"   Combined FPR:           {total_metrics['fpr']:.6f}")

# Save evaluation results
results = {
    "thresholds": {
        "path_a": path_a_threshold,
        "seq_block": seq_block_threshold,
        "seq_review": seq_review_threshold,
    },
    "test_results": {
        "tier1_path_a": tier1_metrics,
        "tier2_path_b_block": tier2_metrics,
        "total_block": total_metrics,
        "tier3_reviews": int(tier3_review.sum()),
        "novel_caught_by_tier2": int(novel_caught_by_tier2),
        "novel_caught_by_review": int(novel_caught_by_review),
        "total_fraud": int(total_fraud),
        "total_missed": int(total_fraud - total_metrics["tp"] - novel_caught_by_review),
    },
}

os.makedirs("experiments/results", exist_ok=True)
with open("experiments/results/v4_three_tier_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n   Results saved → experiments/results/v4_three_tier_results.json")
print(f"   Thresholds saved → {V4_DIR}/")
print("=" * 70)
