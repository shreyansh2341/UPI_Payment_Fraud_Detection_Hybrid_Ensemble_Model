"""
test_hybrid_system.py
─────────────────────
End-to-end test of the Hybrid Fraud Detection System.

Uses REAL PaySim test data for Path A validation (known fraud),
plus synthetic novel fraud for Path B validation.
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

from hybrid_fraud_detector import HybridFraudDetector

MODEL_DIR = Path("models/paysim_v3")
DATA_PATH = Path("data/cleaned_paysim_lstm.csv")
RANDOM_STATE = 42


def load_real_test_data():
    """Load the real PaySim test split (same split as training)."""
    print("  Loading real PaySim data...")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()
    df = df.sort_values("step").reset_index(drop=True)

    # Engineer velocity features
    df["tx_count_cumul"] = np.log1p(df.groupby("nameorig").cumcount() + 1)
    df["amount_cumul"] = np.log1p(df.groupby("nameorig")["amount"].cumsum().clip(0))
    ra = df.groupby("nameorig")["amount"].cumsum() / (df.groupby("nameorig").cumcount() + 1)
    df["amt_vs_avg"] = df["amount"] / (ra + 1e-6)
    df["time_since_last"] = df.groupby("nameorig")["step"].diff().fillna(48).clip(0, 96)
    df["amt_to_bal_ratio"] = np.log1p((df["amount"] / (df["oldbalanceorg"] + 1e-6)).clip(0))
    df["balance_velocity"] = (df["newbalanceorig"] - df["oldbalanceorg"]) / (df["amount"] + 1e-6)

    if "hour" not in df.columns:
        df["hour"] = df["step"] % 24
    if "dayofweek" not in df.columns:
        df["dayofweek"] = (df["step"] // 24) % 7
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)

    features = joblib.load(MODEL_DIR / "paysim_v3_features.pkl")
    base_18 = features[:18]
    for col in base_18:
        df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)

    X = df[base_18]; y = df["isfraud"]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)

    return X_test, y_test


def generate_novel_fraud(n_per_scenario=200):
    """Generate synthetic novel fraud patterns."""
    np.random.seed(42)
    data = []

    # Structuring
    for acc in range(n_per_scenario // 5):
        bal = np.random.uniform(50000, 100000)
        cumul = 0
        for tx in range(5):
            amt = np.random.uniform(8000, 9999)
            new_bal = bal - amt; cumul += amt
            data.append({
                "amount": amt, "oldbalanceorg": bal, "newbalanceorig": new_bal,
                "oldbalancedest": 0, "newbalancedest": 0,
                "hour": np.random.randint(9, 17), "dayofweek": np.random.randint(1, 6),
                "is_weekend": 0, "errorbalanceorig": new_bal + amt - bal,
                "errorbalancedest": 0,
                "upi_type_upi_payment": 1, "upi_type_upi_transfer": 0,
                "tx_count_cumul": np.log1p(tx + 1),
                "amount_cumul": np.log1p(cumul),
                "amt_vs_avg": amt / (cumul / (tx+1) + 1e-6),
                "time_since_last": 0.5 if tx > 0 else 48,
                "amt_to_bal_ratio": np.log1p(max(0, amt / (bal + 1e-6))),
                "balance_velocity": (new_bal - bal) / (amt + 1e-6),
                "scenario": "Structuring"
            })
            bal = new_bal

    # Flow-Through Mules
    for acc in range(n_per_scenario // 5):
        for tx in range(5):
            amt = np.random.uniform(5000, 20000)
            old_bal = amt * np.random.uniform(0.9, 1.1)
            new_bal = old_bal * 0.05
            data.append({
                "amount": amt * 0.90, "oldbalanceorg": old_bal, "newbalanceorig": new_bal,
                "oldbalancedest": 1000, "newbalancedest": 1000 + amt * 0.90,
                "hour": np.random.randint(0, 24), "dayofweek": np.random.randint(1, 8),
                "is_weekend": 1 if np.random.random() > 0.7 else 0,
                "errorbalanceorig": new_bal + amt * 0.90 - old_bal,
                "errorbalancedest": 0,
                "upi_type_upi_payment": 0, "upi_type_upi_transfer": 1,
                "tx_count_cumul": np.log1p(tx + 1),
                "amount_cumul": np.log1p(amt * (tx + 1)),
                "amt_vs_avg": 1.0,
                "time_since_last": np.random.uniform(0.1, 1.0),
                "amt_to_bal_ratio": np.log1p(max(0, amt * 0.90 / (old_bal + 1e-6))),
                "balance_velocity": (new_bal - old_bal) / (amt * 0.90 + 1e-6),
                "scenario": "Flow-Through Mule"
            })

    return pd.DataFrame(data)


def main():
    print("=" * 80)
    print("  HYBRID FRAUD DETECTION SYSTEM -- END-TO-END TEST")
    print("=" * 80)

    detector = HybridFraudDetector()
    print(f"  Block threshold:  {detector.block_threshold:.4f}")
    print(f"  Review threshold: {detector.ae_threshold:.6f}")

    # ═══════════════════════════════════════════════════════
    # TEST 1: REAL PaySim data (Path A validation)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST 1: PATH A -- Real PaySim Data (Auto-Block Performance)")
    print(f"{'='*80}")

    X_test, y_test = load_real_test_data()

    # Sample for speed: 5000 legit + all fraud
    fraud_idx = y_test[y_test == 1].index
    legit_idx = y_test[y_test == 0].sample(5000, random_state=42).index
    sample_idx = fraud_idx.append(legit_idx)
    X_sample = X_test.loc[sample_idx]
    y_sample = y_test.loc[sample_idx]

    print(f"  Sample: {len(X_sample)} rows ({y_sample.sum()} fraud, {(y_sample==0).sum()} legit)")

    results = detector.predict(X_sample)
    decisions = [r["decision"] for r in results]

    blocked = np.array([d == "BLOCK" for d in decisions])
    reviewed = np.array([d == "REVIEW" for d in decisions])
    y = y_sample.values

    tp_block = (blocked & (y == 1)).sum()
    fp_block = (blocked & (y == 0)).sum()
    fn_block = (~blocked & (y == 1)).sum()
    tp_review = (reviewed & (y == 1)).sum()
    fp_review = (reviewed & (y == 0)).sum()

    r_block = tp_block / y.sum() if y.sum() > 0 else 0
    p_block = tp_block / blocked.sum() if blocked.sum() > 0 else 0

    print(f"\n  PATH A (Auto-Block):")
    print(f"    Fraud auto-blocked:  {tp_block}/{y.sum()} ({r_block:.1%})")
    print(f"    False blocks:        {fp_block}/{(y==0).sum()}")
    print(f"    Precision:           {p_block:.1%}")

    print(f"\n  PATH B (Review flags on real data):")
    print(f"    Fraud flagged:       {tp_review}/{y.sum()}")
    print(f"    Legit flagged:       {fp_review}/{(y==0).sum()} ({fp_review/(y==0).sum()*100:.2f}%)")

    combined_caught = tp_block + tp_review
    combined_fp = fp_block + fp_review
    combined_recall = combined_caught / y.sum() if y.sum() > 0 else 0
    combined_prec = combined_caught / (combined_caught + combined_fp) if (combined_caught + combined_fp) > 0 else 0

    print(f"\n  COMBINED (Block + Review):")
    print(f"    Total fraud caught:  {combined_caught}/{y.sum()} ({combined_recall:.1%})")
    print(f"    Total false alarms:  {combined_fp}")
    print(f"    Precision:           {combined_prec:.1%}")

    # ═══════════════════════════════════════════════════════
    # TEST 2: Novel fraud (Path B validation)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST 2: PATH B -- Synthetic Novel Fraud (Review Flagging)")
    print(f"{'='*80}")

    novel_df = generate_novel_fraud(200)
    base_18 = detector.base_18
    novel_results = detector.predict(novel_df[base_18])

    for scenario in novel_df.scenario.unique():
        m = novel_df.scenario == scenario
        sc_results = [novel_results[i] for i in range(len(novel_results)) if m.iloc[i]]
        blocked = sum(1 for r in sc_results if r["decision"] == "BLOCK")
        reviewed = sum(1 for r in sc_results if r["decision"] == "REVIEW")
        allowed = sum(1 for r in sc_results if r["decision"] == "ALLOW")
        total = len(sc_results)
        caught = blocked + reviewed
        print(f"  {scenario:<25}: {caught}/{total} caught ({caught/total*100:.1f}%)  "
              f"[BLOCK={blocked}, REVIEW={reviewed}, ALLOW={allowed}]")

    total_novel = len(novel_results)
    total_caught = sum(1 for r in novel_results if r["decision"] != "ALLOW")
    print(f"\n  Overall novel fraud caught: {total_caught}/{total_novel} ({total_caught/total_novel*100:.1f}%)")

    # ═══════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"  PATH A (Auto-Block):     {r_block:.1%} recall, {p_block:.1%} precision on known fraud")
    print(f"  PATH B (Novel Review):   {total_caught}/{total_novel} ({total_caught/total_novel*100:.1f}%) novel fraud flagged")
    print(f"  Review FP rate (real):   {fp_review/(y==0).sum()*100:.2f}% of legit transactions")
    print(f"  Zero legit blocked:      {fp_block == 0}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
