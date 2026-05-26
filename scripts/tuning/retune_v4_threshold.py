import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import fbeta_score

# ══════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════
DATA_PATH = "data/cleaned_paysim_lstm.csv"
V4_DIR = "models/paysim_v4_experiment"

print("=" * 70)
print(" V4 THRESHOLD TUNING (ON VALIDATION SET)")
print("=" * 70)

# ══════════════════════════════════════════════════════
# 1. LOAD DATA & ENGINEER FEATURES 
# ══════════════════════════════════════════════════════
print("\n📂 Loading data...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower().str.strip()
df = df.sort_values("step").reset_index(drop=True)

print("   Engineering velocity features...")
if "hour" not in df.columns: df["hour"] = df["step"] % 24
if "dayofweek" not in df.columns: df["dayofweek"] = (df["step"] // 24) % 7
if "is_weekend" not in df.columns: df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
for col in ["upi_type_upi_payment", "upi_type_upi_transfer"]:
    if col in df.columns: df[col] = df[col].astype(np.int8)

df["tx_count_cumul"] = df.groupby("nameorig").cumcount() + 1
df["amount_cumul"] = df.groupby("nameorig")["amount"].cumsum()
df["amt_vs_avg"] = df["amount"] / (df["amount_cumul"] / df["tx_count_cumul"] + 1e-6)
df["time_since_last"] = df.groupby("nameorig")["step"].diff().fillna(48).clip(0, 96)
df["amt_to_bal_ratio"] = df["amount"] / (df["oldbalanceorg"] + 1e-6)
df["balance_velocity"] = ((df["newbalanceorig"] - df["oldbalanceorg"]) / (df["amount"] + 1e-6))

df["amount_cumul"] = np.log1p(df["amount_cumul"].clip(0))
df["tx_count_cumul"] = np.log1p(df["tx_count_cumul"])
df["amt_to_bal_ratio"] = np.log1p(df["amt_to_bal_ratio"].clip(0))

TARGET = "isfraud"
y = df[TARGET].values.astype(np.int32)

# Same splits as training
n = len(df)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

# We ONLY need the validation set
df_val = df.iloc[train_end:val_end].reset_index(drop=True)
y_val = df_val[TARGET].values.astype(np.int32)
print(f"   Validation set: {len(df_val):,} transactions ({y_val.sum():,} frauds)")

# ══════════════════════════════════════════════════════
# 2. LOAD V4 MODELS
# ══════════════════════════════════════════════════════
print("\n📦 Loading V4 models...")
# Add src to path for custom components
sys.path.append(os.path.abspath("src"))
from v4_layers import BahdanauAttention, focal_loss

custom_objects = {"BahdanauAttention": BahdanauAttention, "focal_loss": focal_loss}

xgb_m = joblib.load(f"{V4_DIR}/paysim_v4_xgb.pkl")
rf_m = joblib.load(f"{V4_DIR}/paysim_v4_rf.pkl")
ae = tf.keras.models.load_model(f"{V4_DIR}/paysim_v4_ae.keras", compile=False)
seq_model = tf.keras.models.load_model(f"{V4_DIR}/paysim_v4_sequential_winner.keras", custom_objects=custom_objects, compile=False)

scaler_18 = joblib.load(f"{V4_DIR}/paysim_v4_base_scaler.pkl")
scaler_20 = joblib.load(f"{V4_DIR}/paysim_v4_scaler.pkl")
features_18 = joblib.load(f"{V4_DIR}/paysim_v4_features.pkl")

# ══════════════════════════════════════════════════════
# 3. COMPUTE 20 FEATURES FOR VALIDATION SET
# ══════════════════════════════════════════════════════
print("\n⚙️  Computing 20 features for validation set...")

# 18 base
df_val[features_18] = df_val[features_18].fillna(0)
for col in features_18:
    df_val[col] = df_val[col].replace([np.inf, -np.inf], 0)
X_val_18 = df_val[features_18].values.astype(np.float64)
X_val_18_s = scaler_18.transform(X_val_18)

# Feature 19: AE Error
recon = ae.predict(X_val_18_s, batch_size=2048, verbose=0)
ae_err = np.log1p(np.mean(np.square(X_val_18_s - recon), axis=1))

# Feature 20: Sequential Score
SEQ_LENGTH = 5
seq_scores = np.zeros(len(X_val_18_s), dtype=np.float32)

sequences = []
valid_indices = []
for i in range(SEQ_LENGTH, len(X_val_18_s)):
    sequences.append(X_val_18_s[i - SEQ_LENGTH : i])
    valid_indices.append(i)

if len(sequences) > 0:
    sequences = np.array(sequences, dtype=np.float32)
    preds = seq_model.predict(sequences, batch_size=512, verbose=0).ravel()
    seq_scores[valid_indices] = preds

# Scale 20 features
X_val_20 = np.column_stack([X_val_18_s, ae_err, seq_scores])
X_val_20_s = scaler_20.transform(X_val_20)

# ══════════════════════════════════════════════════════
# 4. TUNE THRESHOLD (PATH A)
# ══════════════════════════════════════════════════════
print("\n🎯 Tuning Path A Threshold (XGBoost + RF)...")

xgb_prob = xgb_m.predict_proba(X_val_20_s)[:, 1]
rf_prob = rf_m.predict_proba(X_val_20_s)[:, 1]

# Same weights as training script
base_weight = 0.5
rf_weight = 0.5
ens_prob = (xgb_prob * base_weight) + (rf_prob * rf_weight)

best_f2 = 0
best_thresh = 0.5

# Test thresholds from 0.05 to 0.95
for thresh in np.arange(0.05, 0.96, 0.01):
    preds = (ens_prob >= thresh).astype(int)
    
    tp = ((preds == 1) & (y_val == 1)).sum()
    fp = ((preds == 1) & (y_val == 0)).sum()
    fn = ((preds == 0) & (y_val == 1)).sum()
    
    if (tp + fp) == 0 or (tp + fn) == 0:
        continue
        
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    # F2 heavily favors recall
    f2 = (5 * p * r) / (4 * p + r) if (p + r) > 0 else 0
    
    if f2 > best_f2:
        best_f2 = f2
        best_thresh = thresh

# Save the new threshold
threshold_path = f"{V4_DIR}/paysim_v4_threshold.npy"
old_threshold = float(np.load(threshold_path)[0]) if os.path.exists(threshold_path) else "Unknown"

print(f"\n   ✅ Tuning Complete!")
print(f"   Old (SMOTE) Threshold: {old_threshold:.4f}")
print(f"   New (Valid) Threshold: {best_thresh:.4f}")
print(f"   Validation F2 Score:   {best_f2:.4f}")

np.save(threshold_path, np.array([best_thresh]))
print(f"   Saved new threshold to {threshold_path}")
