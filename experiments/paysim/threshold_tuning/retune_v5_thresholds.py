"""
retune_v5_thresholds.py — V5 Hybrid Threshold Optimization
═══════════════════════════════════════════════════════════
Optimizes Tier 2 (BiLSTM blocking) and Tier 3 (review) thresholds
for the V5 hybrid system.

Key insight: V3's Path A already handles known fraud perfectly.
The BiLSTM's job is to catch truly NOVEL patterns while keeping
false positives extremely low. We optimize for precision on Tier 2
since any FP in a blocking tier is costly.

Usage:
  python experiments/paysim/threshold_tuning/retune_v5_thresholds.py
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

_original_dense_init = tf.keras.layers.Dense.__init__

def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    _original_dense_init(self, *args, **kwargs)

tf.keras.layers.Dense.__init__ = _patched_dense_init

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════
print("=" * 70)
print(" V5 HYBRID THRESHOLD OPTIMIZATION")
print("=" * 70)

DATA_PATH = "data/cleaned_paysim_lstm.csv"
V3_DIR = "models/paysim_v3"
V4_DIR = "models/paysim_v4_experiment"

print(f"\n📂 Loading data...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower().str.strip()
df = df.sort_values("step").reset_index(drop=True)

# Velocity features (same as training)
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

# Time-aware split
n = len(df)
train_end = int(0.70 * n)
val_end = int(0.85 * n)
df_val = df.iloc[train_end:val_end].reset_index(drop=True)
df_test = df.iloc[val_end:].reset_index(drop=True)
y_val = df_val[TARGET].values.astype(np.int32)
y_test = df_test[TARGET].values.astype(np.int32)

print(f"   Val:  {len(y_val):,} txns ({y_val.sum():,} fraud)")
print(f"   Test: {len(y_test):,} txns ({y_test.sum():,} fraud)")


# ══════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════
print(f"\n📦 Loading models...")
from src.v4_layers import BahdanauAttention

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
    "seq_length": joblib.load(f"{V4_DIR}/paysim_v4_seq_length.pkl"),
}
print("   Models loaded ✅")


# ══════════════════════════════════════════════════════
# COMPUTE V3 PATH A ON VALIDATION SET
# ══════════════════════════════════════════════════════
def compute_v3_block(df_split, v3):
    """Compute V3 Path A blocking decisions."""
    v3_base = [f for f in v3["features"] if f != "ae_recon_error"]
    X = np.zeros((len(df_split), len(v3_base)), dtype=np.float64)
    for i, feat in enumerate(v3_base):
        if feat in df_split.columns:
            X[:, i] = df_split[feat].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = v3["scaler"]
    ae_scaler = StandardScaler()
    ae_scaler.mean_ = scaler.mean_[:len(v3_base)]
    ae_scaler.scale_ = scaler.scale_[:len(v3_base)]
    ae_scaler.var_ = scaler.var_[:len(v3_base)]
    ae_scaler.n_features_in_ = len(v3_base)
    ae_scaler.n_samples_seen_ = scaler.n_samples_seen_

    X_s = ae_scaler.transform(X)
    rec = v3["ae"].predict(X_s, batch_size=2048, verbose=0)
    ae_err = np.log1p(np.mean(np.square(X_s - rec), axis=1))

    X_19 = np.column_stack([X, ae_err])
    X_19_s = scaler.transform(X_19)

    prob_xgb = v3["xgb"].predict_proba(X_19_s)[:, 1]
    prob_rf = v3["rf"].predict_proba(X_19_s)[:, 1]
    w = v3["weights"]
    conf = w[0] * prob_xgb + w[1] * prob_rf
    block = conf >= v3["block_threshold"]
    return block, conf


def compute_v4_signals(df_split, v4):
    """Compute V4 BiLSTM signals (sequential scores + anomaly flags)."""
    features = v4["features"]
    X = np.zeros((len(df_split), len(features)), dtype=np.float64)
    for i, feat in enumerate(features):
        if feat in df_split.columns:
            X[:, i] = df_split[feat].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_s = v4["base_scaler"].transform(X)

    # AE errors
    rec = v4["ae"].predict(X_s, batch_size=2048, verbose=0)
    ae_err = np.log1p(np.mean(np.square(X_s - rec), axis=1))

    # Sequential scores
    sl = v4["seq_length"]
    n = len(X_s)
    seq_scores = np.zeros(n, dtype=np.float32)
    seqs, idxs = [], []
    for i in range(sl, n):
        seqs.append(X_s[i - sl : i])
        idxs.append(i)
    if seqs:
        preds = v4["sequential"].predict(
            np.array(seqs, dtype=np.float32), batch_size=512, verbose=0
        ).ravel()
        for idx, pred in zip(idxs, preds):
            seq_scores[idx] = pred

    # Anomaly flags
    ae_flag = ae_err >= v4["ae_threshold"]
    if_flag = v4["iforest"].predict(X_s) == -1
    anomaly = ae_flag | if_flag

    return seq_scores, ae_err, ae_flag, if_flag, anomaly


print("\n🔮 Computing V3 Path A on validation set...")
v3_block_val, v3_conf_val = compute_v3_block(df_val, v3)
print(f"   V3 blocks: {v3_block_val.sum():,} ({(v3_block_val & (y_val == 1)).sum():,} fraud)")

print("\n🔮 Computing V4 signals on validation set...")
val_seq, val_ae, val_ae_flag, val_if_flag, val_anomaly = compute_v4_signals(df_val, v4)
print(f"   AE flags: {val_ae_flag.sum():,}, IForest flags: {val_if_flag.sum():,}")

# Novel fraud = fraud NOT blocked by V3
novel_fraud_val = (y_val == 1) & ~v3_block_val
legit_val = y_val == 0
n_novel = novel_fraud_val.sum()
print(f"   Novel frauds (missed by V3): {n_novel:,}")


# ══════════════════════════════════════════════════════
# OPTIMIZE TIER 2 THRESHOLD
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(" TIER 2 THRESHOLD SWEEP (Novel Fraud Blocking)")
print("=" * 70)

"""
We want the BiLSTM blocking tier to be VERY precise.
Since it's a blocking decision, FPs are costly.

Strategy: find seq_block_threshold that maximizes Precision
while catching as many novel frauds as possible.
Constraint: FPR < 0.5% (very strict for a blocking tier)
"""

print(f"\n   {'SeqThresh':>10} {'NovelBlock':>12} {'NovelRate':>12} {'FP':>8} {'Prec':>10} {'FPR':>10}")
print(f"   {'─' * 64}")

best_thresh = 0.9
best_catch = 0

for t in np.arange(0.10, 0.95, 0.05):
    # Tier 2: seq >= t AND anomaly AND NOT V3-blocked
    tier2 = (val_seq >= t) & val_anomaly & ~v3_block_val

    tier2_tp = (tier2 & (y_val == 1)).sum()
    tier2_fp = (tier2 & legit_val).sum()
    tier2_prec = tier2_tp / (tier2_tp + tier2_fp) if (tier2_tp + tier2_fp) > 0 else 0
    novel_rate = tier2_tp / n_novel if n_novel > 0 else 0
    fpr = tier2_fp / legit_val.sum() if legit_val.sum() > 0 else 0

    marker = ""
    # Maximize novel fraud catch, constraint: FPR < 0.5%
    if fpr < 0.005 and tier2_tp >= best_catch:
        best_catch = tier2_tp
        best_thresh = float(t)
        marker = " ←"

    print(f"   {t:>10.2f} {tier2_tp:>12,} {novel_rate:>12.4f} {tier2_fp:>8,} {tier2_prec:>10.4f} {fpr:>10.6f}{marker}")

# If all novel frauds are already caught by V3, use 99th percentile of legitimate scores
if n_novel == 0 or best_catch == 0:
    best_thresh = float(np.percentile(val_seq[legit_val], 99))
    print(f"\n   V3 catches all fraud. BiLSTM threshold set to 99th pct of legit: {best_thresh:.4f}")

seq_block_threshold = best_thresh
print(f"\n   ✅ Optimized seq_block_threshold: {seq_block_threshold:.4f}")


# ══════════════════════════════════════════════════════
# TIER 3 THRESHOLD (Review)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(" TIER 3 THRESHOLD (Review)")
print("=" * 70)

# 97th percentile of legitimate sequential scores — tighter than before
seq_review_threshold = float(np.percentile(val_seq[legit_val], 97))
print(f"   seq_review_threshold (97th pct legit): {seq_review_threshold:.6f}")


# ══════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(" SAVING THRESHOLDS")
print("=" * 70)

np.save(f"{V4_DIR}/paysim_v4_seq_block_threshold.npy", np.array([seq_block_threshold]))
np.save(f"{V4_DIR}/paysim_v4_seq_threshold.npy", np.array([seq_review_threshold]))

print(f"   seq_block:  {seq_block_threshold:.6f} → paysim_v4_seq_block_threshold.npy")
print(f"   seq_review: {seq_review_threshold:.6f} → paysim_v4_seq_threshold.npy")
print(f"   (V3 block threshold preserved: {v3['block_threshold']:.6f})")


# ══════════════════════════════════════════════════════
# FINAL CHECK ON TEST SET
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(" V5 HYBRID — TEST SET EVALUATION")
print("=" * 70)

print("\n   Computing test set predictions...")
v3_block_test, v3_conf_test = compute_v3_block(df_test, v3)
test_seq, test_ae, test_ae_flag, test_if_flag, test_anomaly = compute_v4_signals(df_test, v4)

# Three-tier decisions
tier1 = v3_block_test
tier2 = (test_seq >= seq_block_threshold) & test_anomaly & ~tier1
tier3 = (test_anomaly | (test_seq >= seq_review_threshold)) & ~tier1 & ~tier2
total_block = tier1 | tier2

total_fraud = y_test.sum()
tier1_tp = (tier1 & (y_test == 1)).sum()
tier1_fp = (tier1 & (y_test == 0)).sum()
tier2_tp = (tier2 & (y_test == 1)).sum()
tier2_fp = (tier2 & (y_test == 0)).sum()
total_tp = (total_block & (y_test == 1)).sum()
total_fp = (total_block & (y_test == 0)).sum()
total_fn = (~total_block & (y_test == 1)).sum()
total_tn = (~total_block & (y_test == 0)).sum()
review_catches = (tier3 & (y_test == 1) & ~tier1 & ~tier2).sum()

recall = total_tp / total_fraud if total_fraud > 0 else 0
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
f1 = 2 * precision * recall / (precision + recall + 1e-12)
fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0

print(f"\n   ═══════════════════════════════════════════")
print(f"   TIER 1 (V3 Path A):      {tier1_tp:>6,} fraud / {tier1_fp:>6,} FP  ({tier1_tp/total_fraud:>6.1%})")
print(f"   TIER 2 (BiLSTM Block):   {tier2_tp:>6,} fraud / {tier2_fp:>6,} FP  ({tier2_tp/total_fraud:>6.1%})")
print(f"   Total blocked:           {total_tp:>6,} fraud / {total_fp:>6,} FP  ({total_tp/total_fraud:>6.1%})")
print(f"   TIER 3 (Review):         {review_catches:>6,} fraud caught")
print(f"   Total reviews:           {tier3.sum():>6,}")
print(f"   Missed entirely:         {total_fraud - total_tp - review_catches:>6,}")
print(f"\n   Recall:    {recall:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   F1:        {f1:.4f}")
print(f"   FPR:       {fpr:.6f}")

# Save final results
results = {
    "thresholds": {
        "v3_block": v3["block_threshold"],
        "seq_block": seq_block_threshold,
        "seq_review": seq_review_threshold,
    },
    "test_results": {
        "tier1": {"tp": int(tier1_tp), "fp": int(tier1_fp)},
        "tier2": {"tp": int(tier2_tp), "fp": int(tier2_fp)},
        "total_block": {
            "tp": int(total_tp), "fp": int(total_fp),
            "recall": float(recall), "precision": float(precision),
            "f1": float(f1), "fpr": float(fpr),
        },
        "tier3": {"fraud_caught": int(review_catches), "total_reviews": int(tier3.sum())},
        "total_fraud": int(total_fraud),
        "total_missed": int(total_fraud - total_tp - review_catches),
    },
}

os.makedirs("experiments/results", exist_ok=True)
with open("experiments/results/v5_optimized_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n   Results → experiments/results/v5_optimized_results.json")
print("=" * 70)
