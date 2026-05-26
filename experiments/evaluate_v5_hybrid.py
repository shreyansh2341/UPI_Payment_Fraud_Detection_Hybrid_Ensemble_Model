"""
evaluate_v5_hybrid.py — V5 Hybrid Evaluation
═════════════════════════════════════════════
Evaluates the V5 hybrid model (V3 Path A + V4 BiLSTM Path B)
against V3 on the same test set.

Shows three-tier breakdown: known fraud blocks, novel fraud blocks, reviews.

Usage:
  python experiments/evaluate_v5_hybrid.py
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

# Keras compatibility patch
_original_dense_init = tf.keras.layers.Dense.__init__

def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    _original_dense_init(self, *args, **kwargs)

tf.keras.layers.Dense.__init__ = _patched_dense_init

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# ══════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════
print("=" * 70)
print(" V5 HYBRID EVALUATION (V3 Path A + V4 BiLSTM Path B)")
print("=" * 70)

DATA_PATH = "data/cleaned_paysim_lstm.csv"
V3_DIR = "models/paysim_v3"
V4_DIR = "models/paysim_v4_experiment"

print(f"\n📂 Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower().str.strip()
df = df.sort_values("step").reset_index(drop=True)

# Engineer velocity features
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

# Time-aware split
n = len(df)
train_end = int(0.70 * n)
val_end = int(0.85 * n)
df_test = df.iloc[val_end:].reset_index(drop=True)

TARGET = "isfraud"
y_test = df_test[TARGET].values.astype(np.int32)

print(f"   Test set: {len(y_test):,} transactions ({y_test.sum():,} frauds)")


# ══════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════
print("\n📦 Loading V3 models...")
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

print("\n📦 Loading V4 models (BiLSTM Path B)...")
from src.v4_layers import BahdanauAttention

seq_block_path = f"{V4_DIR}/paysim_v4_seq_block_threshold.npy"
v4 = {
    "ae": tf.keras.models.load_model(f"{V4_DIR}/paysim_v4_ae.keras", compile=False, safe_mode=False),
    "sequential": tf.keras.models.load_model(
        f"{V4_DIR}/paysim_v4_sequential_winner.keras",
        compile=False, safe_mode=False,
        custom_objects={"BahdanauAttention": BahdanauAttention},
    ),
    "iforest": joblib.load(f"{V4_DIR}/paysim_v4_iforest.pkl"),
    "base_scaler": joblib.load(f"{V4_DIR}/paysim_v4_base_scaler.pkl"),
    "features": joblib.load(f"{V4_DIR}/paysim_v4_features.pkl"),
    "ae_threshold": float(np.load(f"{V4_DIR}/paysim_v4_ae_threshold.npy")[0]),
    "seq_block_threshold": float(np.load(seq_block_path)[0]) if os.path.exists(seq_block_path) else 0.5,
    "seq_threshold": float(np.load(f"{V4_DIR}/paysim_v4_seq_threshold.npy")[0]),
    "seq_length": joblib.load(f"{V4_DIR}/paysim_v4_seq_length.pkl"),
}
print("   V4 loaded ✅")

print(f"\n   Thresholds:")
print(f"     V3 block:        {v3['block_threshold']:.6f}")
print(f"     V4 seq_block:    {v4['seq_block_threshold']:.6f}")
print(f"     V4 seq_review:   {v4['seq_threshold']:.6f}")
print(f"     V4 ae_threshold: {v4['ae_threshold']:.6f}")


# ══════════════════════════════════════════════════════
# TIER 1: V3 PATH A — Known Fraud Auto-Block
# ══════════════════════════════════════════════════════
print("\n🔮 Tier 1: V3 Path A predictions...")

v3_base_features = [f for f in v3["features"] if f != "ae_recon_error"]

X_v3_base = np.zeros((len(df_test), len(v3_base_features)), dtype=np.float64)
for i, feat in enumerate(v3_base_features):
    if feat in df_test.columns:
        X_v3_base[:, i] = df_test[feat].values
X_v3_base = np.nan_to_num(X_v3_base, nan=0.0, posinf=0.0, neginf=0.0)

# V3 scaler for AE
scaler_v3 = v3["scaler"]
v3_ae_scaler = StandardScaler()
v3_ae_scaler.mean_ = scaler_v3.mean_[:len(v3_base_features)]
v3_ae_scaler.scale_ = scaler_v3.scale_[:len(v3_base_features)]
v3_ae_scaler.var_ = scaler_v3.var_[:len(v3_base_features)]
v3_ae_scaler.n_features_in_ = len(v3_base_features)
v3_ae_scaler.n_samples_seen_ = scaler_v3.n_samples_seen_

X_v3_base_s = v3_ae_scaler.transform(X_v3_base)

# V3 AE error
rec_v3 = v3["ae"].predict(X_v3_base_s, batch_size=2048, verbose=0)
ae_err_v3 = np.log1p(np.mean(np.square(X_v3_base_s - rec_v3), axis=1))

# V3 19-feature ensemble
X_v3_19 = np.column_stack([X_v3_base, ae_err_v3])
X_v3_19_s = scaler_v3.transform(X_v3_19)

prob_xgb_v3 = v3["xgb"].predict_proba(X_v3_19_s)[:, 1]
prob_rf_v3 = v3["rf"].predict_proba(X_v3_19_s)[:, 1]
w = v3["weights"]
v3_confidence = w[0] * prob_xgb_v3 + w[1] * prob_rf_v3

# Tier 1: V3 auto-block
tier1_block = v3_confidence >= v3["block_threshold"]

# V3 Path B (review only — for comparison)
v3_ae_flag = ae_err_v3 >= v3["ae_threshold"]
v3_iforest_flag = v3["iforest"].predict(X_v3_base_s) == -1
v3_review = (v3_ae_flag | v3_iforest_flag) & ~tier1_block

print(f"   Tier 1 blocks: {tier1_block.sum():,}")
print(f"   Tier 1 fraud blocked: {(tier1_block & (y_test == 1)).sum():,}/{y_test.sum():,}")


# ══════════════════════════════════════════════════════
# TIER 2 & 3: V4 PATH B — Novel Fraud Detection
# ══════════════════════════════════════════════════════
print("\n🔮 Tier 2/3: V4 Path B predictions (BiLSTM)...")

v4_base_features = v4["features"]
X_v4_base = np.zeros((len(df_test), len(v4_base_features)), dtype=np.float64)
for i, feat in enumerate(v4_base_features):
    if feat in df_test.columns:
        X_v4_base[:, i] = df_test[feat].values
X_v4_base = np.nan_to_num(X_v4_base, nan=0.0, posinf=0.0, neginf=0.0)

X_v4_base_s = v4["base_scaler"].transform(X_v4_base)

# V4 AE errors
rec_v4 = v4["ae"].predict(X_v4_base_s, batch_size=2048, verbose=0)
ae_err_v4 = np.log1p(np.mean(np.square(X_v4_base_s - rec_v4), axis=1))

# Sequential scores (real sequential context)
n_test = len(X_v4_base_s)
seq_scores = np.zeros(n_test, dtype=np.float32)
sequences = []
valid_indices = []
for i in range(v4["seq_length"], n_test):
    sequences.append(X_v4_base_s[i - v4["seq_length"] : i])
    valid_indices.append(i)
if sequences:
    sequences = np.array(sequences, dtype=np.float32)
    preds = v4["sequential"].predict(sequences, batch_size=512, verbose=0).ravel()
    for idx, pred in zip(valid_indices, preds):
        seq_scores[idx] = pred

# Anomaly flags
ae_flag_v4 = ae_err_v4 >= v4["ae_threshold"]
iforest_flag_v4 = v4["iforest"].predict(X_v4_base_s) == -1
anomaly_flag_v4 = ae_flag_v4 | iforest_flag_v4

# Tier 2: Novel fraud blocking
tier2_block = (
    (seq_scores >= v4["seq_block_threshold"])
    & anomaly_flag_v4
    & ~tier1_block
)

# Tier 3: Review (uncertain)
tier3_review = (
    (anomaly_flag_v4 | (seq_scores >= v4["seq_threshold"]))
    & ~tier1_block
    & ~tier2_block
)

# Combined
total_block = tier1_block | tier2_block

print(f"   Tier 2 blocks: {tier2_block.sum():,}")
print(f"   Tier 3 reviews: {tier3_review.sum():,}")


# ══════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(" V5 HYBRID RESULTS")
print("=" * 70)

total_fraud = y_test.sum()
tier1_tp = (tier1_block & (y_test == 1)).sum()
tier1_fp = (tier1_block & (y_test == 0)).sum()
tier2_tp = (tier2_block & (y_test == 1)).sum()
tier2_fp = (tier2_block & (y_test == 0)).sum()
total_tp = (total_block & (y_test == 1)).sum()
total_fp = (total_block & (y_test == 0)).sum()
total_fn = (~total_block & (y_test == 1)).sum()
total_tn = (~total_block & (y_test == 0)).sum()
review_tp = (tier3_review & (y_test == 1) & ~tier1_block & ~tier2_block).sum()

total_recall = total_tp / total_fraud if total_fraud > 0 else 0
total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall + 1e-12)
total_fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0

# V3-only metrics (for comparison)
v3_tp = (tier1_block & (y_test == 1)).sum()
v3_fp = tier1_fp
v3_fn = total_fraud - v3_tp
v3_recall = v3_tp / total_fraud
v3_precision = v3_tp / (v3_tp + v3_fp) if (v3_tp + v3_fp) > 0 else 0
v3_f1 = 2 * v3_precision * v3_recall / (v3_precision + v3_recall + 1e-12)
v3_review_catches = (v3_review & (y_test == 1) & ~tier1_block).sum()

print(f"\n   {'Metric':<25} {'V3 (Path A only)':<18} {'V5 (V3+V4 Hybrid)':<18}")
print(f"   {'─' * 60}")
print(f"   {'Recall (block)':<25} {v3_recall:<18.4f} {total_recall:<18.4f}")
print(f"   {'Precision (block)':<25} {v3_precision:<18.4f} {total_precision:<18.4f}")
print(f"   {'F1 (block)':<25} {v3_f1:<18.4f} {total_f1:<18.4f}")
print(f"   {'FPR (block)':<25} {tier1_fp / (tier1_fp + total_tn + tier2_fp):<18.6f} {total_fpr:<18.6f}")

print(f"\n   V5 THREE-TIER BREAKDOWN:")
print(f"   ═══════════════════════════════════════════════════")
print(f"   Tier 1 (V3 Path A):         {tier1_tp:>6,} fraud blocked / {tier1_fp:>6,} FP  ({tier1_tp/total_fraud:>6.1%})")
print(f"   Tier 2 (V4 BiLSTM Block):   {tier2_tp:>6,} fraud blocked / {tier2_fp:>6,} FP  ({tier2_tp/total_fraud:>6.1%})")
print(f"   Total blocked:              {total_tp:>6,} fraud blocked / {total_fp:>6,} FP  ({total_tp/total_fraud:>6.1%})")
print(f"   Tier 3 (Review):            {review_tp:>6,} fraud in review")
print(f"   Missed entirely:            {total_fraud - total_tp - review_tp:>6,}")

print(f"\n   IMPROVEMENT OVER V3:")
print(f"   ═══════════════════════════════════════════════════")
print(f"   V3 blocks:                  {v3_tp:,} fraud ({v3_recall:.1%})")
print(f"   V5 blocks:                  {total_tp:,} fraud ({total_recall:.1%})")
print(f"   Additional fraud blocked:   +{tier2_tp:,} novel frauds via BiLSTM")
print(f"   V3 review catches:          {v3_review_catches:,}")
print(f"   V5 review catches:          {review_tp:,}")
print(f"   V5 total coverage:          {(total_tp + review_tp)/total_fraud:.1%} (block + review)")

# Save results
results = {
    "v5_hybrid": {
        "tier1_path_a": {"tp": int(tier1_tp), "fp": int(tier1_fp), "recall": float(tier1_tp/total_fraud)},
        "tier2_bilstm_block": {"tp": int(tier2_tp), "fp": int(tier2_fp), "recall": float(tier2_tp/total_fraud)},
        "total_block": {
            "tp": int(total_tp), "fp": int(total_fp), "fn": int(total_fn), "tn": int(total_tn),
            "recall": float(total_recall), "precision": float(total_precision),
            "f1": float(total_f1), "fpr": float(total_fpr),
        },
        "tier3_review": {"fraud_caught": int(review_tp), "total_reviews": int(tier3_review.sum())},
        "total_fraud": int(total_fraud),
        "total_missed": int(total_fraud - total_tp - review_tp),
        "total_coverage": float((total_tp + review_tp) / total_fraud),
    },
    "v3_baseline": {
        "tp": int(v3_tp), "fp": int(v3_fp),
        "recall": float(v3_recall), "precision": float(v3_precision),
        "f1": float(v3_f1),
        "review_catches": int(v3_review_catches),
    },
    "thresholds": {
        "v3_block": v3["block_threshold"],
        "v4_seq_block": v4["seq_block_threshold"],
        "v4_seq_review": v4["seq_threshold"],
        "v4_ae_threshold": v4["ae_threshold"],
    },
}

os.makedirs("experiments/results", exist_ok=True)
with open("experiments/results/v5_hybrid_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n   Results saved → experiments/results/v5_hybrid_results.json")
print("=" * 70)
