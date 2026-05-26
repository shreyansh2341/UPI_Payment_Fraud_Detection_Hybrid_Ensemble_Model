"""
test_unseen_fraud_v3_1.py
─────────────────────────
Tests V3.1 models (with AE error as feature #19) against novel fraud.
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

MODEL_DIR = Path("models/paysim_v3")

BASE_18 = [
    "amount", "oldbalanceorg", "newbalanceorig",
    "oldbalancedest", "newbalancedest",
    "hour", "dayofweek", "is_weekend",
    "errorbalanceorig", "errorbalancedest",
    "upi_type_upi_payment", "upi_type_upi_transfer",
    "tx_count_cumul", "amount_cumul", "amt_vs_avg",
    "time_since_last", "amt_to_bal_ratio", "balance_velocity",
]


def generate_unseen_fraud_data(n=2000):
    np.random.seed(42)
    data = []

    # === Scenario 1: Structuring ===
    for _ in range(60):
        bal = np.random.uniform(50000, 100000)
        cumul = 0
        for tx in range(5):
            amt = np.random.uniform(8000, 9999)
            new_bal = bal - amt
            cumul += amt
            data.append({
                "amount": amt, "oldbalanceorg": bal, "newbalanceorig": new_bal,
                "oldbalancedest": 0, "newbalancedest": 0,
                "hour": np.random.randint(9, 17), "dayofweek": np.random.randint(1, 6),
                "is_weekend": 0, "upi_type_upi_payment": 1, "upi_type_upi_transfer": 0,
                "tx_count_cumul": np.log1p(tx + 1),
                "amount_cumul": np.log1p(cumul),
                "amt_vs_avg": amt / (cumul / (tx+1) + 1e-6),
                "time_since_last": 0.5 if tx > 0 else 48,
                "amt_to_bal_ratio": np.log1p(max(0, amt / (bal + 1e-6))),
                "balance_velocity": (new_bal - bal) / (amt + 1e-6),
                "isFraud": 1, "scenario": "Structuring"
            })
            bal = new_bal

    # === Scenario 2: Flow-Through Mules ===
    for _ in range(60):
        for tx in range(5):
            amt = np.random.uniform(5000, 20000)
            old_bal = amt * np.random.uniform(0.9, 1.1)
            new_bal = old_bal * 0.05
            data.append({
                "amount": amt * 0.90, "oldbalanceorg": old_bal, "newbalanceorig": new_bal,
                "oldbalancedest": 1000, "newbalancedest": 1000 + amt * 0.90,
                "hour": np.random.randint(0, 24),
                "dayofweek": np.random.randint(1, 8),
                "is_weekend": 1 if np.random.random() > 0.7 else 0,
                "upi_type_upi_payment": 0, "upi_type_upi_transfer": 1,
                "tx_count_cumul": np.log1p(tx + 1),
                "amount_cumul": np.log1p(amt * (tx + 1)),
                "amt_vs_avg": 1.0,
                "time_since_last": np.random.uniform(0.1, 1.0),
                "amt_to_bal_ratio": np.log1p(max(0, amt * 0.90 / (old_bal + 1e-6))),
                "balance_velocity": (new_bal - old_bal) / (amt * 0.90 + 1e-6),
                "isFraud": 1, "scenario": "Flow-Through Mule"
            })

    # === Scenario 3: Weekend Late Night ===
    for _ in range(60):
        bal = np.random.uniform(5000, 20000)
        cumul = 0
        for tx in range(5):
            amt = np.random.uniform(500, 2000)
            new_bal = bal - amt
            cumul += amt
            data.append({
                "amount": amt, "oldbalanceorg": bal, "newbalanceorig": new_bal,
                "oldbalancedest": 0, "newbalancedest": 0,
                "hour": np.random.randint(2, 5),
                "dayofweek": np.random.choice([6, 7]),
                "is_weekend": 1, "upi_type_upi_payment": 1, "upi_type_upi_transfer": 0,
                "tx_count_cumul": np.log1p(tx + 1),
                "amount_cumul": np.log1p(cumul),
                "amt_vs_avg": amt / (cumul / (tx+1) + 1e-6),
                "time_since_last": np.random.uniform(0.1, 0.5),
                "amt_to_bal_ratio": np.log1p(max(0, amt / (bal + 1e-6))),
                "balance_velocity": (new_bal - bal) / (amt + 1e-6),
                "isFraud": 1, "scenario": "Weekend Late Night"
            })
            bal = new_bal

    # === Legitimate (control) ===
    n_legit = n - len(data)
    for _ in range(max(0, n_legit)):
        amt = np.random.uniform(10, 500)
        old_bal = np.random.uniform(1000, 10000)
        new_bal = old_bal - amt
        data.append({
            "amount": amt, "oldbalanceorg": old_bal, "newbalanceorig": new_bal,
            "oldbalancedest": 5000, "newbalancedest": 5000 + amt,
            "hour": np.random.randint(8, 22), "dayofweek": np.random.randint(1, 6),
            "is_weekend": 0, "upi_type_upi_payment": 1, "upi_type_upi_transfer": 0,
            "tx_count_cumul": np.log1p(np.random.randint(1, 20)),
            "amount_cumul": np.log1p(np.random.uniform(100, 5000)),
            "amt_vs_avg": np.random.uniform(0.5, 2.0),
            "time_since_last": np.random.uniform(10, 48),
            "amt_to_bal_ratio": np.log1p(max(0, amt / (old_bal + 1e-6))),
            "balance_velocity": (new_bal - old_bal) / (amt + 1e-6),
            "isFraud": 0, "scenario": "Legitimate"
        })

    df = pd.DataFrame(data)
    df["errorbalanceorig"] = df["newbalanceorig"] + df["amount"] - df["oldbalanceorg"]
    df["errorbalancedest"] = df["oldbalancedest"] + df["amount"] - df["newbalancedest"]
    return df


def main():
    print("=" * 80)
    print("  V3.1 STRESS TEST -- AE ERROR AS FEATURE")
    print("=" * 80)

    df = generate_unseen_fraud_data(2000)

    # Load models
    scaler_19 = joblib.load(MODEL_DIR / "paysim_v3_scaler.pkl")
    xgb_m = joblib.load(MODEL_DIR / "paysim_v3_xgb.pkl")
    rf_m = joblib.load(MODEL_DIR / "paysim_v3_rf.pkl")
    thresh = np.load(MODEL_DIR / "paysim_v3_threshold.npy")[0]
    ae = tf.keras.models.load_model(MODEL_DIR / "paysim_v3_ae.keras", compile=False)

    print(f"  Threshold: {thresh:.4f}")
    print(f"  Features: 19 (18 + ae_recon_error)")

    # Get 18-feature matrix
    X_18 = df[BASE_18].values

    # Need to scale 18 features for AE (AE was trained on 18-feature scaled data)
    # We use the first 18 columns of the 19-feature scaler
    from sklearn.preprocessing import StandardScaler
    # Scale 18 features with a temporary scaler (AE needs its own scaling)
    # Actually, we need to compute AE error using the ORIGINAL v3 18-feature scale
    # Since we overwrote the scaler, we'll reconstruct from the 19-feature scaler
    # by using its first 18 components
    mean_18 = scaler_19.mean_[:18]
    scale_18 = scaler_19.scale_[:18]
    X_18_scaled = (X_18 - mean_18) / scale_18

    # Compute AE recon error
    rec = ae.predict(X_18_scaled, batch_size=2048, verbose=0)
    ae_err = np.log1p(np.mean(np.square(X_18_scaled - rec), axis=1))

    # Create 19-feature matrix
    X_19 = np.column_stack([X_18, ae_err.reshape(-1, 1)])
    X_19_scaled = scaler_19.transform(X_19)

    # Predict
    xgb_p = xgb_m.predict_proba(X_19_scaled)[:, 1]
    rf_p = rf_m.predict_proba(X_19_scaled)[:, 1]
    ens_p = 0.5 * xgb_p + 0.5 * rf_p
    preds = (ens_p >= thresh).astype(int)

    y = df["isFraud"].values

    # Per-scenario results
    print()
    print(f"  {'Scenario':<25} | {'Detected':>10} | {'Total':>6} | {'Rate':>8}")
    print(f"  {'-'*25} | {'-'*10} | {'-'*6} | {'-'*8}")
    for sc in df.scenario.unique():
        m = df.scenario == sc
        det = preds[m].sum()
        tot = m.sum()
        rate = det / tot * 100
        label = "TP" if sc != "Legitimate" else "FP"
        print(f"  {sc:<25} | {det:>10} | {tot:>6} | {rate:>6.1f}% {label}")

    # Overall
    tp = ((preds==1) & (y==1)).sum()
    fp = ((preds==1) & (y==0)).sum()
    fn = ((preds==0) & (y==1)).sum()
    tn = ((preds==0) & (y==0)).sum()
    pr = tp/(tp+fp) if (tp+fp)>0 else 0
    rc = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*pr*rc/(pr+rc) if (pr+rc)>0 else 0

    print()
    print(f"  OVERALL FRAUD:")
    print(f"    Recall:    {rc:.1%}  ({tp}/{tp+fn})")
    print(f"    Precision: {pr:.1%}  ({fp} FP)")
    print(f"    F1:        {f1:.4f}")
    print()

    # AE error distribution comparison
    print(f"  AE RECON ERROR (feature #19):")
    for sc in df.scenario.unique():
        m = df.scenario == sc
        e = ae_err[m]
        print(f"    {sc:<22}: mean={np.nanmean(e):.4f}  p50={np.nanmedian(e):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
