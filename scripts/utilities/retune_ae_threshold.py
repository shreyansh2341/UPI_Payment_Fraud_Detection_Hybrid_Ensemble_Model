"""
final_ae_threshold.py
Sets the AE threshold properly and evaluates on BOTH real and synthetic data.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_PATH  = Path("data/cleaned_paysim_lstm.csv")
MODEL_DIR  = Path("models/paysim_v3")
RANDOM_STATE = 42

FEATURES = joblib.load(MODEL_DIR / "paysim_v3_features.pkl")

# ── Load real PaySim data ──
print("Loading real PaySim data...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()
df = df.sort_values("step").reset_index(drop=True)

# Engineer velocity features (same as training)
df["tx_count_cumul"] = np.log1p(df.groupby("nameorig").cumcount() + 1)
df["amount_cumul"] = np.log1p(df.groupby("nameorig")["amount"].cumsum().clip(0))
running_avg = df.groupby("nameorig")["amount"].cumsum() / (df.groupby("nameorig").cumcount() + 1)
df["amt_vs_avg"] = df["amount"] / (running_avg + 1e-6)
df["time_since_last"] = df.groupby("nameorig")["step"].diff().fillna(48).clip(0, 96)
df["amt_to_bal_ratio"] = np.log1p((df["amount"] / (df["oldbalanceorg"] + 1e-6)).clip(0))
df["balance_velocity"] = (df["newbalanceorig"] - df["oldbalanceorg"]) / (df["amount"] + 1e-6)

for col in FEATURES:
    df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)

X = df[FEATURES]; y = df["isfraud"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)

scaler = joblib.load(MODEL_DIR / "paysim_v3_scaler.pkl")
ae = tf.keras.models.load_model(MODEL_DIR / "paysim_v3_ae.keras", compile=False)

X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

print("Computing errors on validation set...")
val_rec = ae.predict(X_val_s, batch_size=2048, verbose=0)
val_err = np.log1p(np.mean(np.square(X_val_s - val_rec), axis=1))

legit_err = val_err[y_val.values == 0]
fraud_err = val_err[y_val.values == 1]

print(f"\nReal PaySim error distribution:")
print(f"  Legit: mean={legit_err.mean():.6f}  p95={np.percentile(legit_err,95):.6f}  p99={np.percentile(legit_err,99):.6f}")
print(f"  Fraud: mean={fraud_err.mean():.6f}  p50={np.median(fraud_err):.6f}")
print(f"  Separation: {fraud_err.mean()/legit_err.mean():.2f}x")

# ── Sweep percentiles on validation ──
print(f"\n{'='*80}")
print(f"  THRESHOLD SWEEP (Real PaySim Validation)")
print(f"{'='*80}")
print(f"  {'Pctile':>8} {'Thresh':>10} | {'Recall':>7} | {'Prec':>7} | {'FP Rate':>7} | {'F1':>7} | {'TP':>5} | {'FP':>6}")

best_f1, best_t = 0, 0

for pct in np.arange(85, 99.95, 0.5):
    thresh = np.percentile(legit_err, pct)
    preds = (val_err >= thresh).astype(int)
    yv = y_val.values

    tp = ((preds==1) & (yv==1)).sum()
    fp = ((preds==1) & (yv==0)).sum()
    fn = ((preds==0) & (yv==1)).sum()
    tn = ((preds==0) & (yv==0)).sum()

    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    if f1 > best_f1:
        best_f1, best_t = f1, thresh

    print(f"  {pct:7.1f}% {thresh:10.6f} | {r:6.1%} | {p:6.1%} | {fpr:6.3%} | {f1:6.4f} | {tp:>5} | {fp:>6}")

print(f"\n  Best F1: {best_f1:.4f} at threshold {best_t:.6f}")

# ── Save the best threshold ──
np.save(MODEL_DIR / "paysim_v3_ae_threshold.npy", np.array([best_t]))
print(f"  SAVED threshold: {best_t:.6f}")

# ── Evaluate on TEST set ──
print(f"\n{'='*80}")
print(f"  TEST SET (threshold={best_t:.6f})")
print(f"{'='*80}")
test_rec = ae.predict(X_test_s, batch_size=2048, verbose=0)
test_err = np.log1p(np.mean(np.square(X_test_s - test_rec), axis=1))
preds = (test_err >= best_t).astype(int)
yt = y_test.values
tp = ((preds==1) & (yt==1)).sum()
fp = ((preds==1) & (yt==0)).sum()
fn = ((preds==0) & (yt==1)).sum()
tn = ((preds==0) & (yt==0)).sum()
p = tp / (tp + fp) if (tp + fp) > 0 else 0
r = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"  Recall:    {r:.1%}  ({tp}/{tp+fn})")
print(f"  Precision: {p:.1%}  ({tp} TP, {fp} FP)")
print(f"  F1:        {f1:.4f}")
print(f"  FP Rate:   {fpr:.3%}")
print(f"  True Neg:  {tn}")
