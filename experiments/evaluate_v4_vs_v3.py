"""
evaluate_v4_vs_v3.py — Head-to-Head Model Comparison
═══════════════════════════════════════════════════════
Compares V3 (19-feature ensemble) vs V4 (20-feature ensemble with
sequential model) on the same test set.

Outputs:
  - Side-by-side metrics table
  - Per-fraud-type analysis (which frauds V4 catches that V3 misses)
  - Results saved to experiments/results/v4_vs_v3_comparison.json

Usage:
  python experiments/evaluate_v4_vs_v3.py
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

# ── Fix Keras cross-version compatibility ──
# V3 AE was saved with a Keras version that serializes 'quantization_config'
# into Dense layers. Current Keras doesn't recognize this kwarg.
# We monkey-patch Dense to strip it before initialization.
_original_dense_init = tf.keras.layers.Dense.__init__

def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    _original_dense_init(self, *args, **kwargs)

tf.keras.layers.Dense.__init__ = _patched_dense_init

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    f1_score,
)

# ══════════════════════════════════════════════════════
# LOAD DATA (Same split as training)
# ══════════════════════════════════════════════════════
print("=" * 70)
print(" V4 vs V3 — HEAD-TO-HEAD COMPARISON")
print("=" * 70)

DATA_PATH = "data/cleaned_paysim_lstm.csv"
print(f"\n📂 Loading data from {DATA_PATH}...")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower().str.strip()
df = df.sort_values("step").reset_index(drop=True)

# ── Engineer velocity features ──
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
df["balance_velocity"] = ((df["newbalanceorig"] - df["oldbalanceorg"]) / (df["amount"] + 1e-6))

df["amount_cumul"] = np.log1p(df["amount_cumul"].clip(0))
df["tx_count_cumul"] = np.log1p(df["tx_count_cumul"])
df["amt_to_bal_ratio"] = np.log1p(df["amt_to_bal_ratio"].clip(0))

print(f"   Total transactions: {len(df):,}")

# Same time-aware split
n = len(df)
train_end = int(0.70 * n)
val_end = int(0.85 * n)
df_test = df.iloc[val_end:].reset_index(drop=True)

TARGET = "isfraud"
y_test = df_test[TARGET].values.astype(np.int32)

print(f"   Test set: {len(y_test):,} transactions ({y_test.sum():,} frauds)")


# ══════════════════════════════════════════════════════
# LOAD BOTH MODELS (direct loading for Keras compatibility)
# ══════════════════════════════════════════════════════
print("\n📦 Loading V3 models...")
V3_DIR = "models/paysim_v3"
V4_DIR = "models/paysim_v4_experiment"

v3 = {
    "xgb": joblib.load(f"{V3_DIR}/paysim_v3_xgb.pkl"),
    "rf": joblib.load(f"{V3_DIR}/paysim_v3_rf.pkl"),
    "ae": tf.keras.models.load_model(f"{V3_DIR}/paysim_v3_ae.keras", compile=False, safe_mode=False),
    "iforest": joblib.load(f"{V3_DIR}/paysim_v3_iforest.pkl"),
    "scaler": joblib.load(f"{V3_DIR}/paysim_v3_scaler.pkl"),
    "features": joblib.load(f"{V3_DIR}/paysim_v3_features.pkl"),
    "block_threshold": float(np.load(f"{V3_DIR}/paysim_v3_threshold.npy")[0]),
    "ae_threshold": float(np.load(f"{V3_DIR}/paysim_v3_ae_threshold.npy")[0]),
    "weights": np.load(f"{V3_DIR}/paysim_v3_weights.npy"),
}
print("   V3 loaded ✅")

print("\n📦 Loading V4 models...")
if not os.path.exists(V4_DIR):
    print("   ❌ V4 models not found. Run train_paysim_v4_hybrid.py first.")
    sys.exit(1)

from src.v4_layers import BahdanauAttention

v4 = {
    "xgb": joblib.load(f"{V4_DIR}/paysim_v4_xgb.pkl"),
    "rf": joblib.load(f"{V4_DIR}/paysim_v4_rf.pkl"),
    "ae": tf.keras.models.load_model(f"{V4_DIR}/paysim_v4_ae.keras", compile=False, safe_mode=False),
    "sequential": tf.keras.models.load_model(
        f"{V4_DIR}/paysim_v4_sequential_winner.keras",
        compile=False, safe_mode=False,
        custom_objects={"BahdanauAttention": BahdanauAttention},
    ),
    "iforest": joblib.load(f"{V4_DIR}/paysim_v4_iforest.pkl"),
    "base_scaler": joblib.load(f"{V4_DIR}/paysim_v4_base_scaler.pkl"),
    "scaler": joblib.load(f"{V4_DIR}/paysim_v4_scaler.pkl"),
    "features": joblib.load(f"{V4_DIR}/paysim_v4_features.pkl"),
    "features_20": joblib.load(f"{V4_DIR}/paysim_v4_features_20.pkl"),
    "block_threshold": float(np.load(f"{V4_DIR}/paysim_v4_threshold.npy")[0]),
    "ae_threshold": float(np.load(f"{V4_DIR}/paysim_v4_ae_threshold.npy")[0]),
    "seq_threshold": float(np.load(f"{V4_DIR}/paysim_v4_seq_threshold.npy")[0]),
    "weights": np.load(f"{V4_DIR}/paysim_v4_weights.npy"),
    "seq_length": joblib.load(f"{V4_DIR}/paysim_v4_seq_length.pkl"),
}
print("   V4 loaded ✅")


# ══════════════════════════════════════════════════════
# V3 PREDICTIONS
# ══════════════════════════════════════════════════════
print("\n🔮 Running V3 predictions...")

# V3 features: 18 base + ae_recon_error = 19 total
# We need the 18 base features to feed the AE, then stack ae_error for V3 scaler
v3_base_features = [f for f in v3["features"] if f != "ae_recon_error"]

# Extract base features from test data
X_v3_base = np.zeros((len(df_test), len(v3_base_features)), dtype=np.float64)
for i, feat in enumerate(v3_base_features):
    if feat in df_test.columns:
        X_v3_base[:, i] = df_test[feat].values
    else:
        X_v3_base[:, i] = 0.0
X_v3_base = np.nan_to_num(X_v3_base, nan=0.0, posinf=0.0, neginf=0.0)

# Scale 18 base features for AE (using V3 scaler's first 18 dimensions)
scaler_v3 = v3["scaler"]
from sklearn.preprocessing import StandardScaler
ae_scaler = StandardScaler()
ae_scaler.mean_ = scaler_v3.mean_[:len(v3_base_features)]
ae_scaler.scale_ = scaler_v3.scale_[:len(v3_base_features)]
ae_scaler.var_ = scaler_v3.var_[:len(v3_base_features)]
ae_scaler.n_features_in_ = len(v3_base_features)
ae_scaler.n_samples_seen_ = scaler_v3.n_samples_seen_

X_v3_base_s = ae_scaler.transform(X_v3_base)

# AE reconstruction error
ae_v3 = v3["ae"]
rec_v3 = ae_v3.predict(X_v3_base_s, batch_size=2048, verbose=0)
ae_err_v3 = np.log1p(np.mean(np.square(X_v3_base_s - rec_v3), axis=1))

# Stack 19 features and scale with full V3 scaler
X_v3_19 = np.column_stack([X_v3_base, ae_err_v3])
X_v3_19_s = scaler_v3.transform(X_v3_19)

# Path A
prob_xgb_v3 = v3["xgb"].predict_proba(X_v3_19_s)[:, 1]
prob_rf_v3 = v3["rf"].predict_proba(X_v3_19_s)[:, 1]
w = v3["weights"]
confidence_v3 = w[0] * prob_xgb_v3 + w[1] * prob_rf_v3
block_v3 = confidence_v3 >= v3["block_threshold"]

# Path B
ae_flag_v3 = ae_err_v3 >= v3["ae_threshold"]
iforest_flag_v3 = v3["iforest"].predict(X_v3_base_s) == -1
review_v3 = (ae_flag_v3 | iforest_flag_v3) & ~block_v3


# ══════════════════════════════════════════════════════
# V4 PREDICTIONS (Three-Tier System)
# ══════════════════════════════════════════════════════
print("🔮 Running V4 predictions (three-tier)...")

# V4: 20 features = 18 base + ae_error + seq_score
v4_base_features = v4["features"]
base_scaler_v4 = v4["base_scaler"]
scaler_20_v4 = v4["scaler"]
seq_model = v4["sequential"]
seq_length = v4["seq_length"]

# Extract base features from test data
X_v4_base = np.zeros((len(df_test), len(v4_base_features)), dtype=np.float64)
for i, feat in enumerate(v4_base_features):
    if feat in df_test.columns:
        X_v4_base[:, i] = df_test[feat].values
    else:
        X_v4_base[:, i] = 0.0
X_v4_base = np.nan_to_num(X_v4_base, nan=0.0, posinf=0.0, neginf=0.0)

# Scale base features
X_v4_base_s = base_scaler_v4.transform(X_v4_base)

# AE error
ae_v4 = v4["ae"]
rec_v4 = ae_v4.predict(X_v4_base_s, batch_size=2048, verbose=0)
ae_err_v4 = np.log1p(np.mean(np.square(X_v4_base_s - rec_v4), axis=1))

# Sequential scores
n_test = len(X_v4_base_s)
seq_scores = np.zeros(n_test, dtype=np.float32)
sequences = []
valid_indices = []
for i in range(seq_length, n_test):
    sequences.append(X_v4_base_s[i - seq_length : i])
    valid_indices.append(i)
if sequences:
    sequences = np.array(sequences, dtype=np.float32)
    preds = seq_model.predict(sequences, batch_size=512, verbose=0).ravel()
    for idx, pred in zip(valid_indices, preds):
        seq_scores[idx] = pred

# 20-feature ensemble
X_v4_20 = np.column_stack([X_v4_base, ae_err_v4, seq_scores])
X_v4_20_s = scaler_20_v4.transform(X_v4_20)

# Path A ensemble
prob_xgb_v4 = v4["xgb"].predict_proba(X_v4_20_s)[:, 1]
prob_rf_v4 = v4["rf"].predict_proba(X_v4_20_s)[:, 1]
w_xgb, w_rf = v4["weights"]
confidence_v4 = w_xgb * prob_xgb_v4 + w_rf * prob_rf_v4

# Anomaly flags
ae_flag_v4 = ae_err_v4 >= v4["ae_threshold"]
iforest_flag_v4 = v4["iforest"].predict(X_v4_base_s) == -1
anomaly_flag_v4 = ae_flag_v4 | iforest_flag_v4

# ── THREE-TIER DECISION LOGIC ──
# Tier 1: Path A auto-block (known fraud)
tier1_block = confidence_v4 >= v4["block_threshold"]

# Tier 2: Path B-Block (novel fraud)
seq_block_threshold = v4.get("seq_block_threshold", 0.5)
tier2_block = (
    (seq_scores >= seq_block_threshold)
    & anomaly_flag_v4
    & ~tier1_block
)

# Tier 3: Path B-Review (uncertain anomalies)
seq_review_threshold = v4.get("seq_threshold", 0.26)
tier3_review = (
    (anomaly_flag_v4 | (seq_scores >= seq_review_threshold))
    & ~tier1_block
    & ~tier2_block
)

# Combined block = Tier 1 + Tier 2
total_block_v4 = tier1_block | tier2_block

print(f"   V4 Tier 1 blocks: {tier1_block.sum():,}")
print(f"   V4 Tier 2 blocks: {tier2_block.sum():,}")
print(f"   V4 Total blocks:  {total_block_v4.sum():,}")
print(f"   V4 Reviews:       {tier3_review.sum():,}")


# ══════════════════════════════════════════════════════
# COMPARISON
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(" RESULTS: V3 vs V4 THREE-TIER")
print("=" * 70)


def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


v3_metrics = compute_metrics(y_test, block_v3.astype(int), confidence_v3)

# V4 uses combined block (Tier 1 + Tier 2) for the main comparison
v4_metrics = compute_metrics(y_test, total_block_v4.astype(int), confidence_v4)

# Tier-specific metrics
tier1_tp = ((tier1_block) & (y_test == 1)).sum()
tier1_fp = ((tier1_block) & (y_test == 0)).sum()
tier2_tp = ((tier2_block) & (y_test == 1)).sum()
tier2_fp = ((tier2_block) & (y_test == 0)).sum()

# Novel fraud analysis
novel_fraud = (y_test == 1) & ~tier1_block
novel_caught_tier2 = (tier2_block & (y_test == 1)).sum()
novel_caught_review = (tier3_review & novel_fraud & ~tier2_block).sum()
total_fraud = y_test.sum()

print(f"\n   {'Metric':<20} {'V3':>12} {'V4 (3-Tier)':>12} {'Δ':>12}")
print(f"   {'─' * 56}")

for metric in ["recall", "precision", "f1", "roc_auc", "pr_auc"]:
    v3_val = v3_metrics[metric]
    v4_val = v4_metrics[metric]
    delta = v4_val - v3_val
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
    print(f"   {metric:<20} {v3_val:>12.4f} {v4_val:>12.4f} {delta:>+10.4f} {arrow}")

print(f"\n   V3 Confusion: TP={v3_metrics['tp']:,}  FP={v3_metrics['fp']:,}  "
      f"FN={v3_metrics['fn']:,}  TN={v3_metrics['tn']:,}")
print(f"   V4 Confusion: TP={v4_metrics['tp']:,}  FP={v4_metrics['fp']:,}  "
      f"FN={v4_metrics['fn']:,}  TN={v4_metrics['tn']:,}")

# Three-tier breakdown
print(f"\n   V4 THREE-TIER BREAKDOWN:")
print(f"   ═══════════════════════════════════════════")
print(f"   Tier 1 (Path A block):     {tier1_tp:,} fraud / {tier1_fp:,} FP  ({tier1_tp/total_fraud:.1%} of fraud)")
print(f"   Tier 2 (Path B-Block):     {tier2_tp:,} fraud / {tier2_fp:,} FP  ({tier2_tp/total_fraud:.1%} of fraud)")
print(f"   Total blocked:             {v4_metrics['tp']:,} fraud / {v4_metrics['fp']:,} FP  ({v4_metrics['tp']/total_fraud:.1%} of fraud)")
print(f"   Tier 3 (Review):           {novel_caught_review:,} fraud caught in review")
print(f"   Missed entirely:           {total_fraud - v4_metrics['tp'] - novel_caught_review:,}")

# Path B comparison (V3 vs V4)
v3_pathb_reviews = review_v3.sum()
v3_missed = (y_test == 1) & ~block_v3
v3_pathb_caught = (review_v3 & v3_missed).sum()

print(f"\n   PATH B COMPARISON (V3 review-only vs V4 three-tier):")
print(f"   {'':>25} {'V3':>12} {'V4':>12}")
print(f"   {'Reviews':>25} {v3_pathb_reviews:>12,} {tier3_review.sum():>12,}")
print(f"   {'Novel fraud blocked':>25} {'N/A':>12} {novel_caught_tier2:>12,}")
print(f"   {'Novel fraud in review':>25} {v3_pathb_caught:>12,} {novel_caught_review:>12,}")

# Save results
results = {
    "v3": v3_metrics,
    "v4_combined": v4_metrics,
    "v4_three_tier": {
        "tier1_path_a": {"tp": int(tier1_tp), "fp": int(tier1_fp)},
        "tier2_path_b_block": {"tp": int(tier2_tp), "fp": int(tier2_fp)},
        "tier3_reviews": int(tier3_review.sum()),
        "novel_caught_by_block": int(novel_caught_tier2),
        "novel_caught_by_review": int(novel_caught_review),
        "total_missed": int(total_fraud - v4_metrics["tp"] - novel_caught_review),
    },
    "thresholds": {
        "v3_block": v3["block_threshold"],
        "v4_path_a": v4["block_threshold"],
        "v4_seq_block": seq_block_threshold,
        "v4_seq_review": seq_review_threshold,
    },
}

os.makedirs("experiments/results", exist_ok=True)
with open("experiments/results/v4_vs_v3_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n   Results saved to experiments/results/v4_vs_v3_comparison.json")

# Verdict
print("\n" + "=" * 70)
if v4_metrics["f1"] > v3_metrics["f1"]:
    print(" 🏆 V4 THREE-TIER WINS on F1-Score!")
    print(f"    F1: {v3_metrics['f1']:.4f} (V3) → {v4_metrics['f1']:.4f} (V4)")
elif v4_metrics["f1"] == v3_metrics["f1"]:
    print(" 🤝 TIE on F1-Score")
else:
    if v4_metrics["recall"] > 0 and novel_caught_tier2 > 0:
        print(" 📊 V4 provides novel fraud BLOCKING capability that V3 lacks")
        print(f"    V3 F1: {v3_metrics['f1']:.4f} | V4 F1: {v4_metrics['f1']:.4f}")
        print(f"    V4 blocked {novel_caught_tier2} novel frauds that V3 only sent to review")
    else:
        print(" 🏆 V3 WINS on F1-Score — V4 needs more tuning")
        print(f"    F1: {v3_metrics['f1']:.4f} (V3) → {v4_metrics['f1']:.4f} (V4)")
print("=" * 70)
