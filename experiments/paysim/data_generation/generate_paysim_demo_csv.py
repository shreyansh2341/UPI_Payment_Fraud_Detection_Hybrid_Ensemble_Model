"""
V3 Data Generator for Frontend Testing
--------------------------------------
Generates synthetic PaySim transactions compatible with the V3 Hybrid Model (18 features).
Includes simulated velocity features to allow realistic testing of the V3 ensemble.

Output: data_generation/paysim_demo_v3.csv
Schema: [amount, oldbalanceorg, ..., balance_velocity] (18 cols)
"""
import numpy as np
import pandas as pd
import os

# =====================================================
# FIX BASE DIRECTORY (MOVE UP FROM data_generation/)
# =====================================================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# =====================================================
# V3 FEATURE SCHEMA (Exact Order)
# =====================================================
PAYSIM_V3_FEATURES = [
    "amount", "oldbalanceorg", "newbalanceorig", "oldbalancedest",
    "newbalancedest", "hour", "dayofweek", "is_weekend",
    "errorbalanceorig", "errorbalancedest",
    "upi_type_upi_payment", "upi_type_upi_transfer",
    # Velocity features (6):
    "tx_count_cumul", "amount_cumul", "amt_vs_avg",
    "time_since_last", "amt_to_bal_ratio", "balance_velocity",
]

# =====================================================
# CONFIG
# =====================================================
N_ROWS = 50
FRAUD_RATIO = 0.04  # 4% fraud

# Use time-based seed for variation
rng = np.random.default_rng()

# =====================================================
# DATA GENERATION
# =====================================================
rows = []

for i in range(N_ROWS):
    # 1. Base Transaction Details
    is_fraud = rng.random() < FRAUD_RATIO
    
    # Amounts
    if is_fraud:
        amount = rng.uniform(20000, 150000)
    else:
        amount = rng.uniform(10, 5000)

    # Origin Balance
    oldbalanceorg = rng.uniform(5000, 200000)
    if is_fraud:
        # Fraud: money disappears without balance update (sometimes)
        # or typical bust-out pattern
        newbalanceorig = oldbalanceorg  # No change despite transfer
        errorbalanceorig = -amount      # Huge mismatch
    else:
        # Normal
        newbalanceorig = max(oldbalanceorg - amount, 0)
        errorbalanceorig = oldbalanceorg - amount - newbalanceorig  # ~0

    # Dest Balance
    oldbalancedest = rng.uniform(0, 50000)
    if is_fraud:
        newbalancedest = oldbalancedest # Money doesn't arrive
        errorbalancedest = amount       # Mismatch
    else:
        newbalancedest = oldbalancedest + amount
        errorbalancedest = 0.0

    # Time Features
    hour = rng.integers(0, 24)
    dayofweek = rng.integers(0, 7)
    is_weekend = 1 if dayofweek >= 5 else 0

    # UPI Type
    is_transfer = rng.random() < 0.6
    upi_type_upi_payment = 0 if is_transfer else 1
    upi_type_upi_transfer = 1 if is_transfer else 0

    # 2. Simulated Velocity Features (Critical for V3)
    # realistic simulation for demo purposes
    
    # tx_count_cumul: History length
    tx_count_cumul = rng.integers(1, 100) if not is_fraud else rng.integers(1, 10)
    
    # amount_cumul: Total spent
    amount_cumul = amount * rng.uniform(1.0, 50.0)
    
    # amt_vs_avg: Current amount relative to history
    # Fraud often spikes this
    if is_fraud:
        amt_vs_avg = rng.uniform(5.0, 20.0) # Huge spike
    else:
        amt_vs_avg = rng.uniform(0.5, 2.0)  # Normal range
        
    # time_since_last: Hours since last tx
    # Fraud bursts often have low time gaps
    if is_fraud:
        time_since_last = rng.exponential(0.5) # Fast
    else:
        time_since_last = rng.exponential(24.0) # Daily/Weekly
        
    # amt_to_bal_ratio: log(amount / (oldbalance + e))
    feature_amt_bal = np.log1p(amount / (oldbalanceorg + 1e-6))
    
    # balance_velocity: rate of balance change
    # (new - old) / amount
    # Normal: (old-amt - old)/amt = -1.0
    # Fraud: (old - old)/amt = 0.0
    balance_velocity = (newbalanceorig - oldbalanceorg) / (amount + 1e-6)

    row = {
        "amount": amount,
        "oldbalanceorg": oldbalanceorg,
        "newbalanceorig": newbalanceorig,
        "oldbalancedest": oldbalancedest,
        "newbalancedest": newbalancedest,
        "hour": hour,
        "dayofweek": dayofweek,
        "is_weekend": is_weekend,
        "errorbalanceorig": errorbalanceorig,
        "errorbalancedest": errorbalancedest,
        "upi_type_upi_payment": upi_type_upi_payment,
        "upi_type_upi_transfer": upi_type_upi_transfer,
        "tx_count_cumul": float(tx_count_cumul),
        "amount_cumul": amount_cumul,
        "amt_vs_avg": amt_vs_avg,
        "time_since_last": time_since_last,
        "amt_to_bal_ratio": feature_amt_bal,
        "balance_velocity": balance_velocity
    }
    rows.append(row)

# =====================================================
# SAVE CSV
# =====================================================
df = pd.DataFrame(rows)[PAYSIM_V3_FEATURES]
OUTPUT_PATH = os.path.join(BASE_DIR, "data_generation", "paysim_demo_v3.csv")
df.to_csv(OUTPUT_PATH, index=False)

print("V3 PaySim Demo CSV generated!")
print(f"Path: {OUTPUT_PATH}")
print(f"Rows: {len(df)}")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")
print("Ready for Frontend testing.")
