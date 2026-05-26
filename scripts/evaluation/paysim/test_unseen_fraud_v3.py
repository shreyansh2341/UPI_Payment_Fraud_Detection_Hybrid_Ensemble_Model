"""
test_unseen_fraud_v3.py
───────────────────────
Tests V3 models (with velocity features) against novel fraud patterns.

Key difference from v2 test: generates ACCOUNT SEQUENCES that include
velocity features, simulating realistic fraud behavior over time.

Usage:
    python test_unseen_fraud_v3.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import tensorflow as tf

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_DIR = Path("models/paysim_v3")

FEATURES = [
    # Original 12
    "amount", "oldbalanceorg", "newbalanceorig",
    "oldbalancedest", "newbalancedest",
    "hour", "dayofweek", "is_weekend",
    "errorbalanceorig", "errorbalancedest",
    "upi_type_upi_payment", "upi_type_upi_transfer",
    # New 6 velocity
    "tx_count_cumul", "amount_cumul", "amt_vs_avg",
    "time_since_last", "amt_to_bal_ratio", "balance_velocity",
]


# ============================================================
# GENERATE NOVEL FRAUD DATA (WITH VELOCITY FEATURES)
# ============================================================
def generate_unseen_fraud_data(n=2000):
    np.random.seed(42)
    data = []

    # ─── Scenario 1: Structuring (Smurfing) ───
    # Account makes 5 rapid transfers of ~9K each
    n_accounts = 60
    for acc in range(n_accounts):
        bal = np.random.uniform(50000, 100000)
        cumul_amount = 0
        for tx in range(5):
            amt = np.random.uniform(8000, 9999)
            new_bal = bal - amt
            cumul_amount += amt

            data.append({
                "amount": amt,
                "oldbalanceorg": bal,
                "newbalanceorig": new_bal,
                "oldbalancedest": 0,
                "newbalancedest": 0,
                "hour": np.random.randint(9, 17),
                "dayofweek": np.random.randint(1, 6),
                "is_weekend": 0,
                "upi_type_upi_payment": 1,
                "upi_type_upi_transfer": 0,
                # VELOCITY FEATURES — this is the key
                "tx_count_cumul": np.log1p(tx + 1),    # 1,2,3,4,5 = high
                "amount_cumul": np.log1p(cumul_amount), # rapidly growing
                "amt_vs_avg": amt / (cumul_amount / (tx+1) + 1e-6),  # ~1.0 (same amounts)
                "time_since_last": 0.5 if tx > 0 else 48,  # Very rapid succession
                "amt_to_bal_ratio": np.log1p(amt / (bal + 1e-6)),  # Moderate to high
                "balance_velocity": (new_bal - bal) / (amt + 1e-6),  # -1.0 (draining)
                "isFraud": 1,
                "scenario": "Structuring"
            })
            bal = new_bal

    # ─── Scenario 2: Flow-Through Mules ───
    # Account receives from 5 sources, then transfers out
    n_accounts_mule = 60
    for acc in range(n_accounts_mule):
        # 5 incoming transfers (looks like mule collection)
        for tx in range(5):
            amt = np.random.uniform(5000, 20000)
            old_bal = amt * np.random.uniform(0.9, 1.1)
            new_bal = old_bal * 0.05  # Nearly drained

            data.append({
                "amount": amt * 0.90,
                "oldbalanceorg": old_bal,
                "newbalanceorig": new_bal,
                "oldbalancedest": 1000,
                "newbalancedest": 1000 + amt * 0.90,
                "hour": np.random.randint(0, 24),
                "dayofweek": np.random.randint(1, 8),
                "is_weekend": 1 if np.random.random() > 0.7 else 0,
                "upi_type_upi_payment": 0,
                "upi_type_upi_transfer": 1,
                # VELOCITY FEATURES
                "tx_count_cumul": np.log1p(tx + 1),
                "amount_cumul": np.log1p(amt * (tx + 1)),
                "amt_vs_avg": 1.0,
                "time_since_last": np.random.uniform(0.1, 1.0),  # Very rapid
                "amt_to_bal_ratio": np.log1p(amt * 0.90 / (old_bal + 1e-6)),  # ~0.8-0.9 (high drain)
                "balance_velocity": (new_bal - old_bal) / (amt * 0.90 + 1e-6),  # Strong negative
                "isFraud": 1,
                "scenario": "Flow-Through Mule"
            })

    # ─── Scenario 3: Weekend Surge ───
    # Rapid late-night weekend transactions
    n_accounts_surge = 60
    for acc in range(n_accounts_surge):
        bal = np.random.uniform(5000, 20000)
        cumul = 0
        for tx in range(5):
            amt = np.random.uniform(500, 2000)
            new_bal = bal - amt
            cumul += amt

            data.append({
                "amount": amt,
                "oldbalanceorg": bal,
                "newbalanceorig": new_bal,
                "oldbalancedest": 0,
                "newbalancedest": 0,
                "hour": np.random.randint(2, 5),  # 2-5am
                "dayofweek": np.random.choice([6, 7]),
                "is_weekend": 1,
                "upi_type_upi_payment": 1,
                "upi_type_upi_transfer": 0,
                # VELOCITY FEATURES
                "tx_count_cumul": np.log1p(tx + 1),
                "amount_cumul": np.log1p(cumul),
                "amt_vs_avg": amt / (cumul / (tx+1) + 1e-6),
                "time_since_last": np.random.uniform(0.1, 0.5),  # Minutes apart
                "amt_to_bal_ratio": np.log1p(amt / (bal + 1e-6)),
                "balance_velocity": (new_bal - bal) / (amt + 1e-6),
                "isFraud": 1,
                "scenario": "Weekend Late Night"
            })
            bal = new_bal

    # ─── Legitimate Transactions (Control) ───
    n_legit = n - len(data)
    for i in range(max(0, n_legit)):
        amt = np.random.uniform(10, 500)
        old_bal = np.random.uniform(1000, 10000)
        new_bal = old_bal - amt

        data.append({
            "amount": amt,
            "oldbalanceorg": old_bal,
            "newbalanceorig": new_bal,
            "oldbalancedest": 5000,
            "newbalancedest": 5000 + amt,
            "hour": np.random.randint(8, 22),
            "dayofweek": np.random.randint(1, 6),
            "is_weekend": 0,
            "upi_type_upi_payment": 1,
            "upi_type_upi_transfer": 0,
            # Normal velocity profile
            "tx_count_cumul": np.log1p(np.random.randint(1, 20)),
            "amount_cumul": np.log1p(np.random.uniform(100, 5000)),
            "amt_vs_avg": np.random.uniform(0.5, 2.0),
            "time_since_last": np.random.uniform(10, 48),  # Hours/days between txns
            "amt_to_bal_ratio": np.log1p(amt / (old_bal + 1e-6)),  # Low ratio
            "balance_velocity": (new_bal - old_bal) / (amt + 1e-6),
            "isFraud": 0,
            "scenario": "Legitimate"
        })

    df = pd.DataFrame(data)

    # Derived features
    df["errorbalanceorig"] = df["newbalanceorig"] + df["amount"] - df["oldbalanceorg"]
    df["errorbalancedest"] = df["oldbalancedest"] + df["amount"] - df["newbalancedest"]

    return df


# ============================================================
# LOAD V3 MODELS
# ============================================================
def load_models():
    print("\n  Loading V3 models...")
    models = {}

    models["xgb"] = joblib.load(MODEL_DIR / "paysim_v3_xgb.pkl")
    models["rf"] = joblib.load(MODEL_DIR / "paysim_v3_rf.pkl")
    models["scaler"] = joblib.load(MODEL_DIR / "paysim_v3_scaler.pkl")
    models["threshold"] = np.load(MODEL_DIR / "paysim_v3_threshold.npy")[0]
    print(f"  [OK] XGB+RF (threshold={models['threshold']:.4f})")

    models["ae"] = tf.keras.models.load_model(MODEL_DIR / "paysim_v3_ae.keras", compile=False)
    models["ae_threshold"] = np.load(MODEL_DIR / "paysim_v3_ae_threshold.npy")[0]
    print(f"  [OK] AE (threshold={models['ae_threshold']:.6f})")

    models["iforest"] = joblib.load(MODEL_DIR / "paysim_v3_iforest.pkl")
    print(f"  [OK] IForest")

    return models


# ============================================================
# PREDICT AND EVALUATE
# ============================================================
def predict_and_evaluate(df, models):
    X = df[FEATURES].values
    X_s = models["scaler"].transform(X)
    y = df["isFraud"].values

    # Layer 1: XGB+RF
    xgb_p = models["xgb"].predict_proba(X_s)[:, 1]
    rf_p = models["rf"].predict_proba(X_s)[:, 1]
    ens_p = 0.5 * xgb_p + 0.5 * rf_p
    l1 = (ens_p >= models["threshold"]).astype(int)

    # Layer 2a: AE
    recon = models["ae"].predict(X_s, batch_size=2048, verbose=0)
    ae_err = np.log1p(np.mean(np.square(X_s - recon), axis=1))
    l2a = (ae_err >= models["ae_threshold"]).astype(int)

    # Layer 2b: IForest
    l2b = (models["iforest"].predict(X_s) == -1).astype(int)

    # Combined OR
    combined = np.maximum(np.maximum(l1, l2a), l2b)

    results = {"XGB+RF": l1, "AE": l2a, "IForest": l2b, "ALL(OR)": combined}

    # Print header
    print(f"\n{'═'*80}")
    print(f"  V3 STRESS TEST — NOVEL FRAUD (WITH VELOCITY FEATURES)")
    print(f"{'═'*80}")

    header = f"  {'Scenario':<22}"
    for name in results:
        header += f" | {name:>10}"
    print(header)
    print(f"  {'─'*len(header)}")

    for sc in df["scenario"].unique():
        mask = df["scenario"] == sc
        row = f"  {sc:<22}"
        for name, preds in results.items():
            detected = preds[mask].sum()
            total = mask.sum()
            rate = detected / total * 100
            if sc == "Legitimate":
                row += f" | {100-rate:>8.1f}%TN"
            else:
                row += f" | {rate:>8.1f}%TP"
        print(row)

    # Overall
    print(f"\n{'─'*80}")
    print(f"  OVERALL (Fraud Only)")
    print(f"{'─'*80}")

    fraud_mask = y == 1
    for name, preds in results.items():
        tp = ((preds==1)&(y==1)).sum()
        fp = ((preds==1)&(y==0)).sum()
        fn = ((preds==0)&(y==1)).sum()
        r = tp/(tp+fn) if (tp+fn)>0 else 0
        p = tp/(tp+fp) if (tp+fp)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        print(f"  {name:<12}: Recall={r:.1%}  Prec={p:.1%}  F1={f1:.4f}  TP={tp} FP={fp}")

    print(f"{'═'*80}\n")


def main():
    df = generate_unseen_fraud_data(2000)
    models = load_models()
    predict_and_evaluate(df, models)

if __name__ == "__main__":
    main()
