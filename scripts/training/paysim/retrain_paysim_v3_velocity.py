"""
retrain_paysim_v3_velocity.py
─────────────────────────────
V3 Models: Adds VELOCITY & BEHAVIORAL features to detect novel fraud.

OPTIMIZED: Uses only vectorized pandas operations — no Python lambdas.

New features engineered per account (using nameOrig/step):
  1. tx_count_cumul     — Cumulative # of transactions from this account
  2. amount_cumul       — Cumulative amount moved by this account
  3. amt_vs_avg         — Current amount / account's running average
  4. time_since_last    — Steps since this account's last transaction
  5. amt_to_bal_ratio   — Amount / sender's balance (how much is being drained)
  6. balance_velocity   — Rate of balance change (newbal - oldbal) / amount

These features capture BEHAVIORAL context that single-transaction
features miss — critical for detecting novel fraud patterns.

Saves to: models/paysim_v3/
Revert to: models/paysim_v2_baseline/

Usage:
    python retrain_paysim_v3_velocity.py
"""

import numpy as np
import pandas as pd
import joblib
import time
import gc
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH  = Path("data/cleaned_paysim_lstm.csv")
MODEL_DIR  = Path("models/paysim_v3")
EVAL_DIR   = Path("evaluation_results/paysim_evaluation_results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "isfraud"


def print_header(text):
    print(f"\n{'═'*70}")
    print(f"  {text}")
    print(f"{'═'*70}\n")


def print_step(num, title, explanation=""):
    print(f"\n{'─'*70}")
    print(f"  STEP {num}: {title}")
    if explanation:
        for line in explanation.split('\n'):
            print(f"  📝 {line.strip()}")
    print(f"{'─'*70}\n")


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
def load_data():
    print_step(1, "LOAD FULL DATASET",
               "Loading cleaned_paysim_lstm.csv (has nameOrig, nameDest, step)")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()
    df = df.sort_values("step").reset_index(drop=True)

    print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Fraud:  {df[TARGET].sum():,} ({df[TARGET].mean()*100:.3f}%)")
    return df


# ============================================================
# STEP 2: ENGINEER VELOCITY FEATURES (VECTORIZED)
# ============================================================
def engineer_velocity_features(df):
    print_step(2, "ENGINEER VELOCITY FEATURES (VECTORIZED)",
               "All operations use cumcount/cumsum/diff — no Python lambdas.\n"
               "This runs in ~30s on 6.36M rows instead of hours.")

    t0 = time.time()

    # Time features
    if "hour" not in df.columns:
        df["hour"] = df["step"] % 24
    if "dayofweek" not in df.columns:
        df["dayofweek"] = (df["step"] // 24) % 7
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)

    for col in ["upi_type_upi_payment", "upi_type_upi_transfer"]:
        if col in df.columns:
            df[col] = df[col].astype(np.int8)

    # --- Feature 1: tx_count (cumulative per-origin account) ---
    print("  Computing tx_count_cumul...")
    df["tx_count_cumul"] = df.groupby("nameorig").cumcount() + 1

    # --- Feature 2: amount_cumul (cumulative amount per-origin) ---
    print("  Computing amount_cumul...")
    df["amount_cumul"] = df.groupby("nameorig")["amount"].cumsum()

    # --- Feature 3: amt_vs_avg (current vs running average) ---
    print("  Computing amt_vs_avg...")
    df["amt_vs_avg"] = df["amount"] / (df["amount_cumul"] / df["tx_count_cumul"] + 1e-6)

    # --- Feature 4: time_since_last (steps since last tx from same account) ---
    print("  Computing time_since_last...")
    df["time_since_last"] = df.groupby("nameorig")["step"].diff().fillna(48)
    df["time_since_last"] = df["time_since_last"].clip(0, 96)

    # --- Feature 5: amt_to_bal_ratio (how much of balance is being moved) ---
    print("  Computing amt_to_bal_ratio...")
    df["amt_to_bal_ratio"] = df["amount"] / (df["oldbalanceorg"] + 1e-6)

    # --- Feature 6: balance_velocity (rate of balance drain) ---
    print("  Computing balance_velocity...")
    df["balance_velocity"] = (
        (df["newbalanceorig"] - df["oldbalanceorg"]) /
        (df["amount"] + 1e-6)
    )

    elapsed = time.time() - t0
    print(f"\n  ✅ Velocity features engineered in {elapsed:.1f}s")

    # Log-transform heavy-tailed features
    df["amount_cumul"] = np.log1p(df["amount_cumul"].clip(0))
    df["tx_count_cumul"] = np.log1p(df["tx_count_cumul"])
    df["amt_to_bal_ratio"] = np.log1p(df["amt_to_bal_ratio"].clip(0))

    return df


# ============================================================
# STEP 3: PREPARE FEATURES AND SPLIT
# ============================================================
def prepare_and_split(df):
    print_step(3, "PREPARE FEATURES AND SPLIT (18 features)")

    FEATURES = [
        # Original 12
        "amount", "oldbalanceorg", "newbalanceorig",
        "oldbalancedest", "newbalancedest",
        "hour", "dayofweek", "is_weekend",
        "errorbalanceorig", "errorbalancedest",
        "upi_type_upi_payment", "upi_type_upi_transfer",
        # New 6 velocity features
        "tx_count_cumul", "amount_cumul", "amt_vs_avg",
        "time_since_last", "amt_to_bal_ratio", "balance_velocity",
    ]

    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    df[FEATURES] = df[FEATURES].fillna(0)

    # Replace inf values
    for col in FEATURES:
        df[col] = df[col].replace([np.inf, -np.inf], 0)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f"  Features: {len(FEATURES)}")
    print(f"  Train: {len(X_train):,} ({y_train.sum():,} fraud)")
    print(f"  Val:   {len(X_val):,}   ({y_val.sum():,} fraud)")
    print(f"  Test:  {len(X_test):,}  ({y_test.sum():,} fraud)")

    joblib.dump(FEATURES, MODEL_DIR / "paysim_v3_features.pkl")
    return X_train, X_val, X_test, y_train, y_val, y_test, FEATURES


# ============================================================
# STEP 4: SCALE
# ============================================================
def scale_features(X_train, X_val, X_test):
    print_step(4, "SCALE FEATURES")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, MODEL_DIR / "paysim_v3_scaler.pkl")
    print(f"  ✅ Scaler saved")
    return X_train_s, X_val_s, X_test_s, scaler


# ============================================================
# STEP 5: TRAIN AUTOENCODER
# ============================================================
def train_autoencoder(X_train_s, y_train):
    print_step(5, "TRAIN AUTOENCODER (18 features, normal-only)",
               "AE now learns velocity norms — abnormal tx_count or\n"
               "amt_vs_avg will cause high reconstruction error.")

    X_normal = X_train_s[y_train.values == 0]
    print(f"  Normal training samples: {len(X_normal):,}")

    input_dim = X_normal.shape[1]
    inp = Input(shape=(input_dim,))
    x = Dense(64, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    x = BatchNormalization()(x)
    encoded = Dense(10, activation="relu")(x)
    x = Dense(32, activation="relu")(encoded)
    x = BatchNormalization()(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    decoded = Dense(input_dim, activation="linear")(x)

    ae = Model(inp, decoded)
    ae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    ae.summary()

    t0 = time.time()
    ae.fit(
        X_normal, X_normal,
        epochs=50, batch_size=1024,
        validation_split=0.1, shuffle=True,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ],
        verbose=1
    )
    elapsed = time.time() - t0
    ae.save(MODEL_DIR / "paysim_v3_ae.keras")
    print(f"\n  ✅ AE trained in {elapsed:.1f}s")
    return ae


# ============================================================
# STEP 6: DERIVE AE THRESHOLD
# ============================================================
def derive_ae_threshold(ae, X_val_s, y_val):
    print_step(6, "DERIVE AE ANOMALY THRESHOLD (F2-optimized)")

    recon = ae.predict(X_val_s, batch_size=2048, verbose=0)
    err = np.log1p(np.mean(np.square(X_val_s - recon), axis=1))

    print(f"  Legit avg error:  {err[y_val.values==0].mean():.6f}")
    print(f"  Fraud avg error:  {err[y_val.values==1].mean():.6f}")
    print(f"  Separation ratio: {err[y_val.values==1].mean()/err[y_val.values==0].mean():.2f}x")

    best_f2, best_thresh = 0, 0
    for pct in np.arange(90, 99.9, 0.1):
        thresh = np.percentile(err[y_val.values == 0], pct)
        preds = (err >= thresh).astype(int)
        tp = ((preds==1)&(y_val.values==1)).sum()
        fp = ((preds==1)&(y_val.values==0)).sum()
        fn = ((preds==0)&(y_val.values==1)).sum()
        p = tp/(tp+fp) if (tp+fp)>0 else 0
        r = tp/(tp+fn) if (tp+fn)>0 else 0
        f2 = (5*p*r/(4*p+r)) if (p+r)>0 else 0
        if f2 > best_f2:
            best_f2, best_thresh = f2, thresh

    np.save(MODEL_DIR / "paysim_v3_ae_threshold.npy", np.array([best_thresh]))
    print(f"  Threshold: {best_thresh:.6f}  F2: {best_f2:.4f}")
    return best_thresh


# ============================================================
# STEP 7: TRAIN ISOLATION FOREST
# ============================================================
def train_iforest(X_train_s, y_train):
    print_step(7, "TRAIN ISOLATION FOREST (18 features)")
    X_normal = X_train_s[y_train.values == 0]

    t0 = time.time()
    iforest = IsolationForest(
        n_estimators=200, contamination=0.002,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    iforest.fit(X_normal)
    elapsed = time.time() - t0
    joblib.dump(iforest, MODEL_DIR / "paysim_v3_iforest.pkl")
    print(f"  ✅ IForest trained in {elapsed:.1f}s")
    return iforest


# ============================================================
# STEP 8: TRAIN XGB + RF
# ============================================================
def train_supervised(X_train_s, y_train, X_val_s, y_val):
    print_step(8, "TRAIN XGB + RF (18 features + SMOTE)")

    print("  Applying SMOTE...")
    smote = SMOTE(sampling_strategy=0.1, random_state=RANDOM_STATE)
    X_sm, y_sm = smote.fit_resample(X_train_s, y_train)
    print(f"  SMOTE: {len(X_sm):,} rows ({y_sm.sum():,} fraud)")

    print("  Training XGBoost...")
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
    print(f"  ✅ XGBoost saved")

    print("  Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=12,
        min_samples_split=10, min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE
    )
    rf_model.fit(X_sm, y_sm)
    joblib.dump(rf_model, MODEL_DIR / "paysim_v3_rf.pkl")
    elapsed = time.time() - t0
    print(f"  ✅ RF saved. Total: {elapsed:.1f}s")

    # Threshold
    print("  Deriving threshold...")
    xgb_p = xgb_model.predict_proba(X_val_s)[:, 1]
    rf_p  = rf_model.predict_proba(X_val_s)[:, 1]
    ens_p = 0.5 * xgb_p + 0.5 * rf_p

    best_f2, best_thresh = 0, 0.5
    for thresh in np.arange(0.01, 0.99, 0.01):
        preds = (ens_p >= thresh).astype(int)
        tp = ((preds==1)&(y_val.values==1)).sum()
        fp = ((preds==1)&(y_val.values==0)).sum()
        fn = ((preds==0)&(y_val.values==1)).sum()
        p = tp/(tp+fp) if (tp+fp)>0 else 0
        r = tp/(tp+fn) if (tp+fn)>0 else 0
        f2 = (5*p*r/(4*p+r)) if (p+r)>0 else 0
        if f2 > best_f2:
            best_f2, best_thresh = f2, thresh

    np.save(MODEL_DIR / "paysim_v3_threshold.npy", np.array([best_thresh]))
    np.save(MODEL_DIR / "paysim_v3_weights.npy", np.array([0.5, 0.5]))
    print(f"  Threshold: {best_thresh:.2f}  F2: {best_f2:.4f}")
    return xgb_model, rf_model, best_thresh


# ============================================================
# STEP 9: EVALUATE ALL ON TEST SET
# ============================================================
def evaluate_all(xgb_m, rf_m, ae, iforest,
                 X_test_s, y_test, xgb_thresh, ae_thresh):
    print_step(9, "EVALUATE ON HELD-OUT TEST SET")

    y = y_test.values

    # Layer 1
    xgb_p = xgb_m.predict_proba(X_test_s)[:, 1]
    rf_p  = rf_m.predict_proba(X_test_s)[:, 1]
    ens_p = 0.5 * xgb_p + 0.5 * rf_p
    l1 = (ens_p >= xgb_thresh).astype(int)

    # Layer 2a: AE
    recon = ae.predict(X_test_s, batch_size=2048, verbose=0)
    ae_err = np.log1p(np.mean(np.square(X_test_s - recon), axis=1))
    l2a = (ae_err >= ae_thresh).astype(int)

    # Layer 2b: IForest
    l2b = (iforest.predict(X_test_s) == -1).astype(int)

    # Combined
    combined = np.maximum(np.maximum(l1, l2a), l2b)

    for name, preds in [("XGB+RF", l1), ("AE", l2a), ("IForest", l2b), ("ALL(OR)", combined)]:
        tp = ((preds==1)&(y==1)).sum()
        fp = ((preds==1)&(y==0)).sum()
        fn = ((preds==0)&(y==1)).sum()
        r = tp/(tp+fn) if (tp+fn)>0 else 0
        p = tp/(tp+fp) if (tp+fp)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        print(f"  {name:<12}: Recall={r:.4f}  Prec={p:.4f}  F1={f1:.4f}  TP={tp} FP={fp} FN={fn}")


# ============================================================
# MAIN
# ============================================================
def main():
    print_header("PAYSIM V3 — VELOCITY FEATURE RETRAINING")

    df = load_data()
    df = engineer_velocity_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test, features = prepare_and_split(df)
    del df; gc.collect()

    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    ae = train_autoencoder(X_train_s, y_train)
    ae_thresh = derive_ae_threshold(ae, X_val_s, y_val)
    iforest = train_iforest(X_train_s, y_train)
    xgb_m, rf_m, xgb_thresh = train_supervised(X_train_s, y_train, X_val_s, y_val)

    evaluate_all(xgb_m, rf_m, ae, iforest,
                 X_test_s, y_test, xgb_thresh, ae_thresh)

    print_header("V3 TRAINING COMPLETE!")
    print(f"  All models saved to: {MODEL_DIR}/")
    print(f"  To revert to v2: copy from models/paysim_v2_baseline/")
    print(f"\n  Next: Run test_unseen_fraud_v3.py")


if __name__ == "__main__":
    main()
