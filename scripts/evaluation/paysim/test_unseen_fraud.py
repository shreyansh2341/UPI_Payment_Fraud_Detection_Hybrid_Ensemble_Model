"""
test_unseen_fraud.py  (v2 — Full 4-Model Stack)
─────────────────────
Tests ALL 4 models against novel fraud patterns:
  Layer 1: XGBoost + Random Forest (Supervised — known patterns)
  Layer 2: Autoencoder + Isolation Forest (Unsupervised — anomaly detection)
  Layer 3: LSTM (Temporal — sequence patterns)

Decision Logic: OR — if ANY layer flags a transaction, it's fraud.

Scenarios Tested:
  1. "Structuring" (Smurfing): Amounts just below threshold
  2. "Flow-Through" Mules: Money in → money out immediately
  3. "Weekend Surge": Rapid late-night weekend transactions

Usage:
    python test_unseen_fraud.py
"""

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf

# ============================================================
# CONFIGURATION
# ============================================================
MODELS_DIR = Path("models/paysim_v2")
OUTPUT_DIR = Path("evaluation_results/paysim_evaluation_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 12 features expected by XGB/RF and AE
FEATURES = [
    "amount", "oldbalanceorg", "newbalanceorig",
    "oldbalancedest", "newbalancedest",
    "hour", "dayofweek", "is_weekend",
    "errorbalanceorig", "errorbalancedest",
    "upi_type_upi_payment", "upi_type_upi_transfer",
]

N_SAMPLES = 2000


# ============================================================
# GENERATE NOVEL FRAUD DATA
# ============================================================
def generate_unseen_fraud_data(n=2000):
    np.random.seed(42)
    data = []

    # ─── Scenario 1: Structuring (Smurfing) ───
    n_structuring = int(n * 0.15)
    for _ in range(n_structuring):
        amt = np.random.uniform(9000, 9999)
        old_bal = np.random.uniform(20000, 50000)
        new_bal = old_bal - amt

        hour = np.random.randint(9, 17)
        day = np.random.randint(1, 6)

        data.append({
            "amount": amt,
            "oldbalanceorg": old_bal,
            "newbalanceorig": new_bal,
            "oldbalancedest": 0,
            "newbalancedest": 0,
            "hour": hour,
            "dayofweek": day,
            "is_weekend": 0,
            "upi_type_upi_payment": 1,
            "upi_type_upi_transfer": 0,
            "isFraud": 1,
            "scenario": "Structuring"
        })

    # ─── Scenario 2: Flow-Through Mules ───
    n_mule = int(n * 0.15)
    for _ in range(n_mule):
        amt = np.random.uniform(5000, 50000)
        old_bal_org = amt
        new_bal_org = amt * 0.10

        day = np.random.randint(1, 8)
        data.append({
            "amount": amt * 0.90,
            "oldbalanceorg": old_bal_org,
            "newbalanceorig": new_bal_org,
            "oldbalancedest": 1000,
            "newbalancedest": 1000 + (amt * 0.90),
            "hour": np.random.randint(0, 24),
            "dayofweek": day,
            "is_weekend": 1 if day > 5 else 0,
            "upi_type_upi_payment": 0,
            "upi_type_upi_transfer": 1,
            "isFraud": 1,
            "scenario": "Flow-Through Mule"
        })

    # ─── Scenario 3: Weekend Surge ───
    n_surge = int(n * 0.15)
    for _ in range(n_surge):
        amt = np.random.uniform(500, 2000)
        old_bal = np.random.uniform(5000, 20000)
        new_bal = old_bal - amt

        hour = np.random.randint(2, 5)
        day = np.random.choice([6, 7])

        data.append({
            "amount": amt,
            "oldbalanceorg": old_bal,
            "newbalanceorig": new_bal,
            "oldbalancedest": 0,
            "newbalancedest": 0,
            "hour": hour,
            "dayofweek": day,
            "is_weekend": 1,
            "upi_type_upi_payment": 1,
            "upi_type_upi_transfer": 0,
            "isFraud": 1,
            "scenario": "Weekend Late Night"
        })

    # ─── Legitimate Transactions (Control Group) ───
    n_legit = n - n_structuring - n_mule - n_surge
    for _ in range(n_legit):
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
            "isFraud": 0,
            "scenario": "Legitimate"
        })

    df = pd.DataFrame(data)

    # Derived features
    df["errorbalanceorig"] = df["newbalanceorig"] + df["amount"] - df["oldbalanceorg"]
    df["errorbalancedest"] = df["oldbalancedest"] + df["amount"] - df["newbalancedest"]

    return df


# ============================================================
# LOAD ALL MODELS
# ============================================================
def load_all_models():
    print("\n  Loading all 4 models from models/paysim_v2/...")

    models = {}

    # Layer 1: XGB + RF
    try:
        models["xgb"] = joblib.load(MODELS_DIR / "paysim_xgb_stage2.pkl")
        models["rf"] = joblib.load(MODELS_DIR / "paysim_rf_stage2.pkl")
        models["xgb_rf_scaler"] = joblib.load(MODELS_DIR / "paysim_stage2_scaler.pkl")
        models["xgb_rf_threshold"] = np.load(MODELS_DIR / "paysim_stage2_threshold.npy")[0]
        print(f"  ✅ Layer 1: XGB + RF loaded (threshold={models['xgb_rf_threshold']:.4f})")
    except Exception as e:
        print(f"  ❌ Layer 1 failed: {e}")

    # Layer 2: AE + IForest
    try:
        models["ae"] = tf.keras.models.load_model(
            MODELS_DIR / "paysim_ae_model_v2.keras", compile=False
        )
        models["ae_scaler"] = joblib.load(MODELS_DIR / "paysim_ae_scaler.pkl")
        models["ae_threshold"] = np.load(MODELS_DIR / "paysim_ae_threshold.npy")[0]
        models["iforest"] = joblib.load(MODELS_DIR / "paysim_iforest_v2.pkl")
        print(f"  ✅ Layer 2: AE + IForest loaded (ae_thresh={models['ae_threshold']:.6f})")
    except Exception as e:
        print(f"  ❌ Layer 2 failed: {e}")

    # Layer 3: LSTM (can't easily use on single transactions — skip for now)
    # LSTM needs sequences of 5 transactions, but our stress test generates
    # individual transactions. We'll note this limitation.
    print(f"  ⚠️  Layer 3: LSTM requires sequences — evaluated separately")

    return models


# ============================================================
# PREDICT WITH ALL LAYERS
# ============================================================
def predict_all_layers(df, models):
    X = df[FEATURES].values

    results = {}

    # --- Layer 1: XGB + RF (Supervised) ---
    if "xgb" in models and "rf" in models:
        X_l1 = models["xgb_rf_scaler"].transform(X)
        xgb_probs = models["xgb"].predict_proba(X_l1)[:, 1]
        rf_probs = models["rf"].predict_proba(X_l1)[:, 1]
        ens_probs = 0.5 * xgb_probs + 0.5 * rf_probs
        l1_preds = (ens_probs >= models["xgb_rf_threshold"]).astype(int)
        results["Layer1_XGB_RF"] = l1_preds
        df["score_xgb_rf"] = ens_probs

    # --- Layer 2a: Autoencoder (Anomaly Detection) ---
    if "ae" in models:
        X_l2 = models["ae_scaler"].transform(X)
        recon = models["ae"].predict(X_l2, batch_size=2048, verbose=0)
        mse = np.mean(np.square(X_l2 - recon), axis=1)
        ae_error = np.log1p(mse)
        ae_preds = (ae_error >= models["ae_threshold"]).astype(int)
        results["Layer2a_AE"] = ae_preds
        df["score_ae"] = ae_error

    # --- Layer 2b: Isolation Forest ---
    if "iforest" in models:
        X_l2 = models["ae_scaler"].transform(X)
        if_raw = models["iforest"].predict(X_l2)
        if_preds = (if_raw == -1).astype(int)
        results["Layer2b_IForest"] = if_preds
        df["score_iforest"] = -models["iforest"].score_samples(X_l2)

    # --- Combined: OR-logic ---
    all_preds = np.zeros(len(df), dtype=int)
    for name, preds in results.items():
        all_preds = np.maximum(all_preds, preds)
    results["COMBINED_OR"] = all_preds

    return results


# ============================================================
# EVALUATE AND PRINT
# ============================================================
def evaluate_scenarios(df, results):
    y_true = df["isFraud"].values
    scenarios = df["scenario"].unique()

    print(f"\n{'═'*80}")
    print(f"  FULL STACK STRESS TEST — NOVEL FRAUD PATTERNS ({len(df)} transactions)")
    print(f"{'═'*80}")

    # Print header
    layer_names = list(results.keys())
    header = f"  {'Scenario':<22}"
    for name in layer_names:
        short = name.replace("Layer", "L").replace("COMBINED_OR", "ALL(OR)")
        header += f" | {short:>10}"
    print(header)
    print(f"  {'-'*len(header)}")

    # Per-scenario results
    for scenario in scenarios:
        mask = df["scenario"] == scenario
        subset_y = y_true[mask]
        row = f"  {scenario:<22}"

        for name, preds in results.items():
            detected = preds[mask].sum()
            total = mask.sum()

            if scenario == "Legitimate":
                # For legit: detected = false positives
                fp_rate = (detected / total * 100)
                row += f" | {100-fp_rate:>8.1f}%TN"
            else:
                recall = (detected / total * 100)
                row += f" | {recall:>8.1f}%TP"

        print(row)

    # Overall stats
    print(f"\n{'─'*80}")
    print(f"  OVERALL METRICS (Novel Fraud Only)")
    print(f"{'─'*80}")

    fraud_mask = y_true == 1
    for name, preds in results.items():
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        tn = ((preds == 0) & (y_true == 0)).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0

        print(f"  {name:<22}: Recall={recall:.1%}  Prec={precision:.1%}  "
              f"F1={f1:.4f}  TP={tp} FP={fp}")

    print(f"{'═'*80}\n")

    # Save
    results_path = OUTPUT_DIR / "unseen_fraud_full_stack_results.csv"
    for name, preds in results.items():
        df[f"pred_{name}"] = preds
    df.to_csv(results_path, index=False)
    print(f"  ✅ Detailed results saved to: {results_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    # 1. Generate novel fraud data
    df = generate_unseen_fraud_data(N_SAMPLES)

    # 2. Load all models
    models = load_all_models()

    # 3. Predict with all layers
    results = predict_all_layers(df, models)

    # 4. Evaluate
    evaluate_scenarios(df, results)


if __name__ == "__main__":
    main()
