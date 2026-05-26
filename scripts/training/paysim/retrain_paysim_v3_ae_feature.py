"""
retrain_paysim_v3_ae_feature.py
────────────────────────────────
V3.1: Uses AE reconstruction error as FEATURE #19 in XGB/RF.

Instead of OR-logic (AE makes its own binary decision → FP explosion),
the AE becomes a feature extractor. XGB/RF learns WHEN high AE error
actually correlates with fraud — dramatically improving precision.

Pipeline:
  1. Load data, engineer velocity features (same 18 as v3)
  2. Load trained AE from v3
  3. Compute reconstruction error for every row
  4. Add ae_recon_error as feature #19
  5. Retrain XGB + RF with 19 features + SMOTE
  6. Evaluate: precision should jump from ~20% to 90%+

Saves to: models/paysim_v3/ (overwrites XGB+RF, keeps AE)
"""

import numpy as np
import pandas as pd
import joblib
import time
import gc
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
DATA_PATH  = Path("data/cleaned_paysim_lstm.csv")
MODEL_DIR  = Path("models/paysim_v3")
RANDOM_STATE = 42
TARGET = "isfraud"

def p(text):
    print(f"  {text}")

def header(text):
    print(f"\n{'='*70}\n  {text}\n{'='*70}")

def step(n, title):
    print(f"\n{'~'*70}\n  STEP {n}: {title}\n{'~'*70}")


# ============================================================
# STEP 1: LOAD + ENGINEER VELOCITY (same as v3)
# ============================================================
header("PAYSIM V3.1 -- AE ERROR AS FEATURE")

step(1, "LOAD DATA + ENGINEER VELOCITY FEATURES")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()
df = df.sort_values("step").reset_index(drop=True)
p(f"Loaded: {len(df):,} rows, {df[TARGET].sum():,} fraud")

t0 = time.time()
df["tx_count_cumul"] = np.log1p(df.groupby("nameorig").cumcount() + 1)
df["amount_cumul"] = np.log1p(df.groupby("nameorig")["amount"].cumsum().clip(0))
running_avg = df.groupby("nameorig")["amount"].cumsum() / (df.groupby("nameorig").cumcount() + 1)
df["amt_vs_avg"] = df["amount"] / (running_avg + 1e-6)
df["time_since_last"] = df.groupby("nameorig")["step"].diff().fillna(48).clip(0, 96)
df["amt_to_bal_ratio"] = np.log1p((df["amount"] / (df["oldbalanceorg"] + 1e-6)).clip(0))
df["balance_velocity"] = (df["newbalanceorig"] - df["oldbalanceorg"]) / (df["amount"] + 1e-6)
p(f"Velocity features engineered in {time.time()-t0:.1f}s")

# Ensure time features exist
if "hour" not in df.columns:
    df["hour"] = df["step"] % 24
if "dayofweek" not in df.columns:
    df["dayofweek"] = (df["step"] // 24) % 7
if "is_weekend" not in df.columns:
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)

# ============================================================
# STEP 2: PREPARE BASE 18 FEATURES + SPLIT
# ============================================================
step(2, "SPLIT DATA (same split as v3)")

BASE_FEATURES = [
    "amount", "oldbalanceorg", "newbalanceorig",
    "oldbalancedest", "newbalancedest",
    "hour", "dayofweek", "is_weekend",
    "errorbalanceorig", "errorbalancedest",
    "upi_type_upi_payment", "upi_type_upi_transfer",
    "tx_count_cumul", "amount_cumul", "amt_vs_avg",
    "time_since_last", "amt_to_bal_ratio", "balance_velocity",
]

for col in BASE_FEATURES:
    df[col] = df[col].fillna(0).replace([np.inf, -np.inf], 0)

X = df[BASE_FEATURES]
y = df[TARGET]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)
p(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# ============================================================
# STEP 3: SCALE WITH V3 SCALER + COMPUTE AE ERRORS
# ============================================================
step(3, "LOAD V3 AE + COMPUTE RECONSTRUCTION ERROR")

scaler_18 = joblib.load(MODEL_DIR / "paysim_v3_scaler.pkl")
ae = tf.keras.models.load_model(MODEL_DIR / "paysim_v3_ae.keras", compile=False)

X_train_s18 = scaler_18.transform(X_train)
X_val_s18   = scaler_18.transform(X_val)
X_test_s18  = scaler_18.transform(X_test)

p("Computing AE reconstruction errors...")
t0 = time.time()

train_rec = ae.predict(X_train_s18, batch_size=4096, verbose=0)
train_ae_err = np.log1p(np.mean(np.square(X_train_s18 - train_rec), axis=1))

val_rec = ae.predict(X_val_s18, batch_size=4096, verbose=0)
val_ae_err = np.log1p(np.mean(np.square(X_val_s18 - val_rec), axis=1))

test_rec = ae.predict(X_test_s18, batch_size=4096, verbose=0)
test_ae_err = np.log1p(np.mean(np.square(X_test_s18 - test_rec), axis=1))

p(f"Done in {time.time()-t0:.1f}s")
p(f"Train AE error: legit={train_ae_err[y_train.values==0].mean():.6f}  "
  f"fraud={train_ae_err[y_train.values==1].mean():.6f}  "
  f"ratio={train_ae_err[y_train.values==1].mean()/train_ae_err[y_train.values==0].mean():.2f}x")

# ============================================================
# STEP 4: CREATE 19-FEATURE DATASET
# ============================================================
step(4, "CREATE 19-FEATURE DATASET (18 + ae_recon_error)")

FEATURES_19 = BASE_FEATURES + ["ae_recon_error"]

X_train_19 = np.column_stack([X_train.values, train_ae_err.reshape(-1, 1)])
X_val_19   = np.column_stack([X_val.values, val_ae_err.reshape(-1, 1)])
X_test_19  = np.column_stack([X_test.values, test_ae_err.reshape(-1, 1)])

# Scale all 19 features
scaler_19 = StandardScaler()
X_train_s19 = scaler_19.fit_transform(X_train_19)
X_val_s19   = scaler_19.transform(X_val_19)
X_test_s19  = scaler_19.transform(X_test_19)

p(f"Feature count: {len(FEATURES_19)}")
p(f"Features: {FEATURES_19}")

joblib.dump(FEATURES_19, MODEL_DIR / "paysim_v3_features.pkl")
joblib.dump(scaler_19, MODEL_DIR / "paysim_v3_scaler.pkl")
p("Saved 19-feature list and scaler (overwrites v3 18-feature versions)")

# ============================================================
# STEP 5: SMOTE + TRAIN XGB
# ============================================================
step(5, "SMOTE + TRAIN XGBOOST (19 features)")

smote = SMOTE(sampling_strategy=0.1, random_state=RANDOM_STATE)
X_sm, y_sm = smote.fit_resample(X_train_s19, y_train)
p(f"SMOTE: {len(X_sm):,} rows ({y_sm.sum():,} fraud)")

t0 = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, gamma=0.1,
    reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=(y_sm==0).sum()/(y_sm==1).sum(),
    eval_metric="aucpr", tree_method="hist",
    random_state=RANDOM_STATE
)
xgb_model.fit(X_sm, y_sm)
joblib.dump(xgb_model, MODEL_DIR / "paysim_v3_xgb.pkl")
p(f"XGBoost trained in {time.time()-t0:.1f}s")

# ============================================================
# STEP 6: TRAIN RANDOM FOREST (19 features)
# ============================================================
step(6, "TRAIN RANDOM FOREST (19 features)")

t0 = time.time()
rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=12,
    min_samples_split=10, min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1, random_state=RANDOM_STATE
)
rf_model.fit(X_sm, y_sm)
joblib.dump(rf_model, MODEL_DIR / "paysim_v3_rf.pkl")
p(f"Random Forest trained in {time.time()-t0:.1f}s")

# ============================================================
# STEP 7: DERIVE THRESHOLD ON VALIDATION
# ============================================================
step(7, "DERIVE OPTIMAL THRESHOLD (F2-optimized)")

xgb_p = xgb_model.predict_proba(X_val_s19)[:, 1]
rf_p  = rf_model.predict_proba(X_val_s19)[:, 1]
ens_p = 0.5 * xgb_p + 0.5 * rf_p

best_f2, best_thresh = 0, 0.5
for thresh in np.arange(0.01, 0.99, 0.01):
    preds = (ens_p >= thresh).astype(int)
    yv = y_val.values
    tp = ((preds==1) & (yv==1)).sum()
    fp = ((preds==1) & (yv==0)).sum()
    fn = ((preds==0) & (yv==1)).sum()
    pr = tp/(tp+fp) if (tp+fp)>0 else 0
    rc = tp/(tp+fn) if (tp+fn)>0 else 0
    f2 = (5*pr*rc/(4*pr+rc)) if (pr+rc)>0 else 0
    if f2 > best_f2:
        best_f2, best_thresh = f2, thresh

np.save(MODEL_DIR / "paysim_v3_threshold.npy", np.array([best_thresh]))
np.save(MODEL_DIR / "paysim_v3_weights.npy", np.array([0.5, 0.5]))
p(f"Threshold: {best_thresh:.2f}  F2: {best_f2:.4f}")

# ============================================================
# STEP 8: FULL EVALUATION ON TEST SET
# ============================================================
step(8, "EVALUATE ON TEST SET")

xgb_p = xgb_model.predict_proba(X_test_s19)[:, 1]
rf_p  = rf_model.predict_proba(X_test_s19)[:, 1]
ens_p = 0.5 * xgb_p + 0.5 * rf_p
preds = (ens_p >= best_thresh).astype(int)

yt = y_test.values
tp = ((preds==1) & (yt==1)).sum()
fp = ((preds==1) & (yt==0)).sum()
fn = ((preds==0) & (yt==1)).sum()
tn = ((preds==0) & (yt==0)).sum()
precision = tp/(tp+fp) if (tp+fp)>0 else 0
recall = tp/(tp+fn) if (tp+fn)>0 else 0
f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
fpr = fp/(fp+tn) if (fp+tn)>0 else 0

header("V3.1 TEST SET RESULTS")
p(f"Recall:     {recall:.4f}  ({tp}/{tp+fn} fraud caught)")
p(f"Precision:  {precision:.4f}  ({fp} false positives)")
p(f"F1 Score:   {f1:.4f}")
p(f"FP Rate:    {fpr:.4%}")
p(f"Threshold:  {best_thresh:.2f}")
p(f"True Neg:   {tn}")
p(f"")
p(f"Feature #19 (ae_recon_error) importance in XGB:")

# Feature importance
importances = xgb_model.feature_importances_
for i, feat in enumerate(FEATURES_19):
    if importances[i] > 0.02 or feat == "ae_recon_error":
        p(f"  {feat:25s}: {importances[i]:.4f}")

header("V3.1 TRAINING COMPLETE")
p(f"Models saved to: {MODEL_DIR}/")
p(f"XGB+RF now use 19 features (18 + ae_recon_error)")
p(f"No more OR-logic -- AE is a feature, not a separate detector")
p(f"")
p(f"Next: python test_unseen_fraud_v3.py (update for 19 features)")
