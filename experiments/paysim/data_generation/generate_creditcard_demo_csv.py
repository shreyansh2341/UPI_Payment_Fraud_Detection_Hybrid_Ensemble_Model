import numpy as np
import pandas as pd
import joblib
import os

# =====================================================
# FIX BASE DIRECTORY
# =====================================================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

FEATURE_PATH = os.path.join(BASE_DIR, "models", "cc_features.pkl")
cc_features = joblib.load(FEATURE_PATH)

# =====================================================
# CONFIG
# =====================================================
N_ROWS = 50
FRAUD_RATIO = 0.15

rng = np.random.default_rng(123)

# =====================================================
# DATA GENERATION
# =====================================================
data = []

for _ in range(N_ROWS):
    is_fraud = rng.random() < FRAUD_RATIO

    row = {
        f: rng.normal(5, 2) if is_fraud else rng.normal(0, 1)
        for f in cc_features
    }
    
   # ADD GROUND TRUTH LABEL FOR VALIDATION
    row['is_fraud'] = int(is_fraud)

    data.append(row)

# =====================================================
# SAVE CSV
# =====================================================
# Keep features in same order, add is_fraud at end
output_columns = cc_features + ['is_fraud']
df = pd.DataFrame(data)[output_columns]
OUTPUT_PATH = os.path.join(BASE_DIR, "data_generation/creditcard_demo_final.csv")
df.to_csv(OUTPUT_PATH, index=False)

actual_fraud_count = df['is_fraud'].sum()

print("✅ Credit Card demo CSV generated")
print("📄 Path:", OUTPUT_PATH)
print("Rows:", len(df))
print(f"Actual frauds: {actual_fraud_count} / {N_ROWS} ({actual_fraud_count/N_ROWS*100:.1f}%)")
