"""
generate_synthetic_frauds.py
────────────────────────────
Generates a balanced training CSV: 5 000 legit + 1 800 fraud rows.
Six distinct fraud strategies so the model learns diverse attack patterns.

Run FIRST, then run  retrain_stage2.py  on the output.

    python generate_synthetic_frauds.py
    →  data/paysim_augmented_with_frauds.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED            = 42
N_LEGIT         = 5_000
N_FRAUD_EACH    = 300          # x 6 strategies  =  1 800 frauds

# Base directory for the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Training output
OUTPUT_TRAIN = PROJECT_ROOT / "data" / "paysim_augmented_with_frauds.csv"
OUTPUT_TRAIN.parent.mkdir(exist_ok=True)

# Frontend Test CSVs output
OUTPUT_TEST_DIR = PROJECT_ROOT / "csv_files"
OUTPUT_TEST_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)

# ─── column order (must match Stage-2 training) ──────────────
COLS = [
    "amount", "oldbalanceorg", "newbalanceorig",
    "oldbalancedest", "newbalancedest",
    "hour", "dayofweek", "is_weekend",
    "errorbalanceorig", "errorbalancedest",
    "has_balance_mismatch",
    "upi_type_upi_payment", "upi_type_upi_transfer",
    "isFraud",
]


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def _hour(n):
    return np.random.randint(0, 24, size=n)

def _dow(n):
    return np.random.randint(0, 7, size=n)

def _upi_flags(n):
    pay = np.random.binomial(1, 0.4, size=n)
    return pay, 1 - pay

def _to_df(d, label):
    d["isFraud"] = label
    return pd.DataFrame(d, columns=COLS)


# ══════════════════════════════════════════════════════════════
# LEGIT  (clean math, errors = 0)
# ══════════════════════════════════════════════════════════════
def make_legit(n):
    amt  = np.clip(np.random.lognormal(7.5, 1.2, n), 10, 50_000)
    ob_o = amt + np.clip(np.random.lognormal(9, 1, n), 100, 2_000_000)
    nb_o = ob_o - amt
    ob_d = np.clip(np.random.lognormal(9, 1, n), 100, 2_000_000)
    nb_d = ob_d + amt
    h, d = _hour(n), _dow(n)
    pay, tra = _upi_flags(n)

    return _to_df({
        "amount": amt,
        "oldbalanceorg": ob_o,   "newbalanceorig": nb_o,
        "oldbalancedest": ob_d,  "newbalancedest": nb_d,
        "hour": h, "dayofweek": d, "is_weekend": (d >= 5).astype(int),
        "errorbalanceorig": ob_o - amt - nb_o,
        "errorbalancedest": ob_d + amt - nb_d,
        "has_balance_mismatch": np.zeros(n, dtype=int),
        "upi_type_upi_payment": pay,
        "upi_type_upi_transfer": tra,
    }, label=0)


# ══════════════════════════════════════════════════════════════
# FRAUD STRATEGIES
# ══════════════════════════════════════════════════════════════

def fraud_A_balance_zeroing(n):
    """Entire origin balance drained. amount == oldbalanceOrig, newbalanceOrig == 0."""
    ob_o = np.clip(np.random.lognormal(10, 0.8, n), 5_000, 500_000)
    amt  = ob_o.copy()
    nb_o = np.zeros(n)
    ob_d = np.clip(np.random.lognormal(9, 1, n), 100, 2_000_000)
    nb_d = ob_d + amt
    h, d = _hour(n), _dow(n)

    return _to_df({
        "amount": amt,
        "oldbalanceorg": ob_o,   "newbalanceorig": nb_o,
        "oldbalancedest": ob_d,  "newbalancedest": nb_d,
        "hour": h, "dayofweek": d, "is_weekend": (d >= 5).astype(int),
        "errorbalanceorig": ob_o - amt - nb_o,
        "errorbalancedest": ob_d + amt - nb_d,
        "has_balance_mismatch": np.zeros(n, dtype=int),
        "upi_type_upi_payment": np.zeros(n, dtype=int),
        "upi_type_upi_transfer": np.ones(n, dtype=int),
    }, label=1)


def fraud_B_dest_frozen(n):
    """Origin debited correctly, dest balance never changes. errorBalanceDest == amount."""
    amt  = np.clip(np.random.lognormal(9.5, 0.7, n), 2_000, 200_000)
    ob_o = amt + np.clip(np.random.lognormal(9, 1, n), 100, 2_000_000)
    nb_o = ob_o - amt
    ob_d = np.clip(np.random.lognormal(9, 1, n), 100, 2_000_000)
    nb_d = ob_d.copy()                              # frozen
    h, d = _hour(n), _dow(n)
    err_o = ob_o - amt - nb_o
    err_d = ob_d + amt - nb_d                       # == amt

    return _to_df({
        "amount": amt,
        "oldbalanceorg": ob_o,   "newbalanceorig": nb_o,
        "oldbalancedest": ob_d,  "newbalancedest": nb_d,
        "hour": h, "dayofweek": d, "is_weekend": (d >= 5).astype(int),
        "errorbalanceorig": err_o,
        "errorbalancedest": err_d,
        "has_balance_mismatch": ((np.abs(err_o) > 1) | (np.abs(err_d) > 1)).astype(int),
        "upi_type_upi_payment": np.zeros(n, dtype=int),
        "upi_type_upi_transfer": np.ones(n, dtype=int),
    }, label=1)


def fraud_C_origin_inflated(n):
    """newbalanceOrig barely drops. errorBalanceOrig is large negative."""
    amt   = np.clip(np.random.lognormal(9.5, 0.7, n), 2_000, 200_000)
    ob_o  = amt + np.clip(np.random.lognormal(9, 1, n), 100, 2_000_000)
    nb_o  = ob_o - amt * np.random.uniform(0.0, 0.2, size=n)   # barely drops
    ob_d  = np.clip(np.random.lognormal(9, 1, n), 100, 2_000_000)
    nb_d  = ob_d + amt
    h, d  = _hour(n), _dow(n)
    err_o = ob_o - amt - nb_o
    err_d = ob_d + amt - nb_d

    return _to_df({
        "amount": amt,
        "oldbalanceorg": ob_o,   "newbalanceorig": nb_o,
        "oldbalancedest": ob_d,  "newbalancedest": nb_d,
        "hour": h, "dayofweek": d, "is_weekend": (d >= 5).astype(int),
        "errorbalanceorig": err_o,
        "errorbalancedest": err_d,
        "has_balance_mismatch": ((np.abs(err_o) > 1) | (np.abs(err_d) > 1)).astype(int),
        "upi_type_upi_payment": np.zeros(n, dtype=int),
        "upi_type_upi_transfer": np.ones(n, dtype=int),
    }, label=1)


def fraud_D_late_night_large(n):
    """Clean math (errors = 0). Signal is large amount at 0-3 AM only."""
    amt  = np.random.uniform(40_000, 180_000, size=n)
    ob_o = amt + np.random.uniform(1_000, 50_000, size=n)
    nb_o = ob_o - amt
    ob_d = np.clip(np.random.lognormal(9, 1, n), 100, 2_000_000)
    nb_d = ob_d + amt
    h    = np.random.randint(0, 4, size=n)          # 0-3 AM
    d    = _dow(n)

    return _to_df({
        "amount": amt,
        "oldbalanceorg": ob_o,   "newbalanceorig": nb_o,
        "oldbalancedest": ob_d,  "newbalancedest": nb_d,
        "hour": h, "dayofweek": d, "is_weekend": (d >= 5).astype(int),
        "errorbalanceorig": ob_o - amt - nb_o,
        "errorbalancedest": ob_d + amt - nb_d,
        "has_balance_mismatch": np.zeros(n, dtype=int),
        "upi_type_upi_payment": np.zeros(n, dtype=int),
        "upi_type_upi_transfer": np.ones(n, dtype=int),
    }, label=1)


def fraud_E_split_structuring(n):
    """Moderate amount but origin balance barely above it (account nearly drained)."""
    amt  = np.random.uniform(3_000, 18_000, size=n)
    ob_o = amt * np.random.uniform(1.01, 1.3, size=n)
    nb_o = ob_o - amt
    ob_d = np.clip(np.random.lognormal(9, 1, n), 100, 2_000_000)
    nb_d = ob_d + amt
    h, d = _hour(n), _dow(n)

    return _to_df({
        "amount": amt,
        "oldbalanceorg": ob_o,   "newbalanceorig": nb_o,
        "oldbalancedest": ob_d,  "newbalancedest": nb_d,
        "hour": h, "dayofweek": d, "is_weekend": (d >= 5).astype(int),
        "errorbalanceorig": ob_o - amt - nb_o,
        "errorbalancedest": ob_d + amt - nb_d,
        "has_balance_mismatch": np.zeros(n, dtype=int),
        "upi_type_upi_payment": np.zeros(n, dtype=int),
        "upi_type_upi_transfer": np.ones(n, dtype=int),
    }, label=1)


def fraud_F_double_spend(n):
    """Origin debited twice, dest credited once. errorBalanceOrig = -amount."""
    amt  = np.clip(np.random.lognormal(9.5, 0.7, n), 2_000, 200_000)
    ob_o = amt * 2 + np.clip(np.random.lognormal(8, 1, n), 100, 500_000)
    nb_o = ob_o - amt * 2
    ob_d = np.clip(np.random.lognormal(9, 1, n), 100, 2_000_000)
    nb_d = ob_d + amt
    h, d = _hour(n), _dow(n)
    err_o = ob_o - amt - nb_o                       # = -amt
    err_d = ob_d + amt - nb_d

    return _to_df({
        "amount": amt,
        "oldbalanceorg": ob_o,   "newbalanceorig": nb_o,
        "oldbalancedest": ob_d,  "newbalancedest": nb_d,
        "hour": h, "dayofweek": d, "is_weekend": (d >= 5).astype(int),
        "errorbalanceorig": err_o,
        "errorbalancedest": err_d,
        "has_balance_mismatch": ((np.abs(err_o) > 1) | (np.abs(err_d) > 1)).astype(int),
        "upi_type_upi_payment": np.zeros(n, dtype=int),
        "upi_type_upi_transfer": np.ones(n, dtype=int),
    }, label=1)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parts = [
        ("Legit",              make_legit(N_LEGIT)),
        ("A - Balance Zero",   fraud_A_balance_zeroing(N_FRAUD_EACH)),
        ("B - Dest Frozen",    fraud_B_dest_frozen(N_FRAUD_EACH)),
        ("C - Origin Inflated",fraud_C_origin_inflated(N_FRAUD_EACH)),
        ("D - Late Night",     fraud_D_late_night_large(N_FRAUD_EACH)),
        ("E - Structuring",    fraud_E_split_structuring(N_FRAUD_EACH)),
        ("F - Double Spend",   fraud_F_double_spend(N_FRAUD_EACH)),
    ]

    for name, df in parts:
        print(f"  {name:25s}  ->  {len(df):>5d} rows  "
              f"(fraud = {int(df['isFraud'].sum())})")

    full = pd.concat([df for _, df in parts], ignore_index=True)
    full = full.sample(frac=1, random_state=SEED).reset_index(drop=True)

    total  = len(full)
    frauds = int(full["isFraud"].sum())
    print(f"\n{'─'*50}")
    print(f"  Total   : {total}")
    print(f"  Frauds  : {frauds}   ({frauds / total * 100:.1f} %)")
    print(f"  Legit   : {total - frauds}")
    print(f"{'─'*50}")

    full.to_csv(OUTPUT_TRAIN, index=False)
    print(f"\nSaved Training Data ->  {OUTPUT_TRAIN}")

    # --- Generate Test CSV Files for Frontend ---
    print("\nGenerating Test CSV files for frontend...")
    
    # 1. Normal batch (95% legit, 5% fraud) - 200 rows
    legit_test = full[full['isFraud'] == 0].sample(190, random_state=SEED)
    fraud_test = full[full['isFraud'] == 1].sample(10, random_state=SEED)
    test_1 = pd.concat([legit_test, fraud_test]).sample(frac=1, random_state=SEED)
    test_1.to_csv(OUTPUT_TEST_DIR / "test_normal_traffic.csv", index=False)
    print(f"  Generated: csv_files/test_normal_traffic.csv (200 rows, 10 fraud)")

    # 2. Fraud heavy batch (50% legit, 50% fraud) - 100 rows
    legit_test_2 = full[full['isFraud'] == 0].sample(50, random_state=SEED+1)
    fraud_test_2 = full[full['isFraud'] == 1].sample(50, random_state=SEED+1)
    test_2 = pd.concat([legit_test_2, fraud_test_2]).sample(frac=1, random_state=SEED+1)
    test_2.to_csv(OUTPUT_TEST_DIR / "test_fraud_heavy.csv", index=False)
    print(f"  Generated: csv_files/test_fraud_heavy.csv (100 rows, 50 fraud)")

    # 3. Clean batch (100% legit) - 100 rows
    test_3 = full[full['isFraud'] == 0].sample(100, random_state=SEED+2)
    test_3.to_csv(OUTPUT_TEST_DIR / "test_clean_legit.csv", index=False)
    print(f"  Generated: csv_files/test_clean_legit.csv (100 rows, 0 fraud)")

    # 4. Tiny test (10 rows, 2 fraud) - Handpicked for quick scan test
    legit_test_4 = full[full['isFraud'] == 0].sample(8, random_state=SEED+3)
    fraud_test_4 = full[full['isFraud'] == 1].sample(2, random_state=SEED+3)
    test_4 = pd.concat([legit_test_4, fraud_test_4]).sample(frac=1, random_state=SEED+3)
    test_4.to_csv(OUTPUT_TEST_DIR / "test_tiny_quick.csv", index=False)
    print(f"  Generated: csv_files/test_tiny_quick.csv (10 rows, 2 fraud)")

    print(f"\nAll test files successfully saved to: {OUTPUT_TEST_DIR}")


if __name__ == "__main__":
    main()