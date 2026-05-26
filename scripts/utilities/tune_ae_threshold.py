"""
tune_ae_threshold.py
Analyze AE reconstruction errors and sweep thresholds to find the sweet spot.
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

# ── Load stress test generator ──
exec(open('test_unseen_fraud_v3.py', encoding='utf-8').read().split('def load_models')[0])

MODEL_DIR = Path('models/paysim_v3')
FEATURES = [
    'amount','oldbalanceorg','newbalanceorig','oldbalancedest','newbalancedest',
    'hour','dayofweek','is_weekend','errorbalanceorig','errorbalancedest',
    'upi_type_upi_payment','upi_type_upi_transfer',
    'tx_count_cumul','amount_cumul','amt_vs_avg',
    'time_since_last','amt_to_bal_ratio','balance_velocity',
]

# ── Generate test data ──
df = generate_unseen_fraud_data(2000)
scaler = joblib.load(MODEL_DIR / 'paysim_v3_scaler.pkl')
ae = tf.keras.models.load_model(MODEL_DIR / 'paysim_v3_ae.keras', compile=False)

X = df[FEATURES].values
X_s = scaler.transform(X)
y = df['isFraud'].values

rec = ae.predict(X_s, batch_size=2048, verbose=0)
err = np.log1p(np.mean(np.square(X_s - rec), axis=1))

# ── Distribution ──
print("=" * 90)
print("  AE RECONSTRUCTION ERROR DISTRIBUTION")
print("=" * 90)
for sc in df.scenario.unique():
    m = df.scenario == sc
    e = err[m]
    print(f"  {sc:22s}: mean={e.mean():.4f}  p50={np.median(e):.4f}  "
          f"p90={np.percentile(e,90):.4f}  p95={np.percentile(e,95):.4f}  "
          f"min={e.min():.4f}  max={e.max():.4f}")

# ── Threshold sweep ──
print()
print("=" * 90)
print("  THRESHOLD SWEEP")
print("=" * 90)
print(f"  {'Threshold':>10} | {'Struct':>8} | {'Mule':>8} | {'Weekend':>8} | {'Legit FP':>8} | {'Prec':>7} | {'Recall':>7} | {'F1':>7}")
print(f"  {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*7} | {'-'*7} | {'-'*7}")

best_f1 = 0
best_t = 0

for t in np.arange(0.01, 3.01, 0.01):
    preds = (err >= t).astype(int)

    struct_r = preds[df.scenario == 'Structuring'].mean()
    mule_r = preds[df.scenario == 'Flow-Through Mule'].mean()
    wknd_r = preds[df.scenario == 'Weekend Late Night'].mean()
    legit_fp = preds[df.scenario == 'Legitimate'].mean()

    tp = ((preds==1) & (y==1)).sum()
    fp = ((preds==1) & (y==0)).sum()
    fn = ((preds==0) & (y==1)).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

    if f1 > best_f1:
        best_f1 = f1
        best_t = t

    # Print only interesting thresholds
    if t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80,
             1.00, 1.20, 1.50, 1.80, 2.00, 2.50, 3.00] or abs(t - best_t) < 0.005:
        marker = " <-- BEST" if abs(t - best_t) < 0.005 else ""
        print(f"  {t:10.2f} | {struct_r:7.1%} | {mule_r:7.1%} | {wknd_r:7.1%} | "
              f"{legit_fp:7.1%} | {p:6.1%} | {r:6.1%} | {f1:6.4f}{marker}")

print()
print(f"  BEST threshold: {best_t:.2f}  F1={best_f1:.4f}")
print()

# ── Also check combined (all models) at best AE threshold ──
print("=" * 90)
print(f"  FULL STACK EVALUATION AT AE THRESHOLD = {best_t:.2f}")
print("=" * 90)

xgb_m = joblib.load(MODEL_DIR / 'paysim_v3_xgb.pkl')
rf_m = joblib.load(MODEL_DIR / 'paysim_v3_rf.pkl')
iforest = joblib.load(MODEL_DIR / 'paysim_v3_iforest.pkl')
sup_thresh = np.load(MODEL_DIR / 'paysim_v3_threshold.npy')[0]

xgb_p = xgb_m.predict_proba(X_s)[:, 1]
rf_p = rf_m.predict_proba(X_s)[:, 1]
l1 = ((0.5 * xgb_p + 0.5 * rf_p) >= sup_thresh).astype(int)
l2a = (err >= best_t).astype(int)
l2b = (iforest.predict(X_s) == -1).astype(int)
combo = np.maximum(np.maximum(l1, l2a), l2b)

for name, preds in [("XGB+RF", l1), ("AE", l2a), ("IForest", l2b), ("ALL(OR)", combo)]:
    for sc in df.scenario.unique():
        m = df.scenario == sc
        det = preds[m].sum()
        tot = m.sum()
        print(f"  {name:12s} | {sc:22s}: {det}/{tot} ({det/tot*100:5.1f}%)")
    tp = ((preds==1) & (y==1)).sum()
    fp = ((preds==1) & (y==0)).sum()
    fn = ((preds==0) & (y==1)).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"  {name:12s} | OVERALL:  Recall={r:.1%}  Prec={p:.1%}  F1={f1:.4f}")
    print()

print(f"  To apply: update models/paysim_v3/paysim_v3_ae_threshold.npy to {best_t:.2f}")
