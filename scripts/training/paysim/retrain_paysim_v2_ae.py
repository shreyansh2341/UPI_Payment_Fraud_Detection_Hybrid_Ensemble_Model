"""
retrain_paysim_v2_ae.py
───────────────────────
Trains Layer 2 (Anomaly Detection) of the anti-overfitting stack:
  - Autoencoder: Learns to reconstruct NORMAL transactions.
    Anything it can't reconstruct well → anomaly → potential fraud.
  - Isolation Forest: Statistical outlier detector.
    Points that are "easy to isolate" with random splits → anomaly.

Both models are UNSUPERVISED — they never see fraud labels during training.
This is the key advantage: they catch fraud types they've never seen before.

Saves to: models/paysim_v2/
  - paysim_ae_model_v2.keras
  - paysim_ae_threshold.npy
  - paysim_iforest_v2.pkl
  - paysim_ae_scaler.pkl

Usage:
    python retrain_paysim_v2_ae.py
"""

import numpy as np
import pandas as pd
import joblib
import time
import gc
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix,
    classification_report, f1_score
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = Path("data/cleaned_paysim.csv")
MODEL_DIR = Path("models/paysim_v2")
EVAL_DIR  = Path("evaluation_results/paysim_evaluation_results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Same 12 features as XGB/RF (no leakage)
FEATURES = [
    "amount", "oldbalanceorg", "newbalanceorig",
    "oldbalancedest", "newbalancedest",
    "hour", "dayofweek", "is_weekend",
    "errorbalanceorig", "errorbalancedest",
    "upi_type_upi_payment", "upi_type_upi_transfer",
]

TARGET = "isfraud"

# AE Hyperparameters
AE_EPOCHS     = 50
AE_BATCH      = 1024
AE_LATENT_DIM = 8    # Bottleneck size (forces compression)

# Isolation Forest
IF_N_ESTIMATORS  = 200
IF_CONTAMINATION = 0.002  # Expected anomaly rate (~0.13% fraud + margin)

RANDOM_STATE = 42


def print_header(text):
    print(f"\n{'═'*70}")
    print(f"  {text}")
    print(f"{'═'*70}\n")


def print_step(num, title, explanation=""):
    print(f"\n{'─'*70}")
    print(f"  STEP {num}: {title}")
    if explanation:
        print(f"  📝 {explanation}")
    print(f"{'─'*70}\n")


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
def load_data():
    print_step(1, "LOAD FULL DATASET",
               "Loading 6.36M rows from cleaned_paysim.csv")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()

    print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Fraud:  {df[TARGET].sum():,} ({df[TARGET].mean()*100:.3f}%)")

    return df


# ============================================================
# STEP 2: SPLIT DATA (same seed as XGB/RF for consistency)
# ============================================================
def split_data(df):
    print_step(2, "STRATIFIED DATA SPLIT (70/15/15)",
               "Using the SAME random seed as XGB/RF for consistency")

    X = df[FEATURES]
    y = df[TARGET]

    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    # Second split: 50/50 of the 30% → 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f"  Train: {len(X_train):,} ({y_train.sum():,} fraud)")
    print(f"  Val:   {len(X_val):,}   ({y_val.sum():,} fraud)")
    print(f"  Test:  {len(X_test):,}  ({y_test.sum():,} fraud)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# STEP 3: SCALE FEATURES
# ============================================================
def scale_features(X_train, X_val, X_test):
    print_step(3, "SCALE FEATURES",
               "StandardScaler fitted on training data only")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # Save AE-specific scaler
    joblib.dump(scaler, MODEL_DIR / "paysim_ae_scaler.pkl")
    print(f"  ✅ Scaler saved to {MODEL_DIR / 'paysim_ae_scaler.pkl'}")

    return X_train_s, X_val_s, X_test_s, scaler


# ============================================================
# STEP 4: TRAIN AUTOENCODER (NORMAL TRANSACTIONS ONLY)
# ============================================================
def train_autoencoder(X_train_s, y_train):
    print_step(4, "TRAIN AUTOENCODER",
               "Training on LEGITIMATE transactions only.\n"
               "     The AE learns 'what normal looks like'.\n"
               "     When fraud comes in → poor reconstruction → high error → flagged.")

    # Extract only legitimate transactions
    X_normal = X_train_s[y_train.values == 0]
    print(f"  Normal training samples: {len(X_normal):,}")
    print(f"  (Excluded {(y_train.values == 1).sum():,} fraud samples from AE training)")

    input_dim = X_normal.shape[1]

    # Architecture: Deeper with batch normalization for better generalization
    inp = Input(shape=(input_dim,), name="ae_input")

    # Encoder
    x = Dense(64, activation="relu", name="enc_1")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu", name="enc_2")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    encoded = Dense(AE_LATENT_DIM, activation="relu", name="bottleneck")(x)

    # Decoder (mirror of encoder)
    x = Dense(32, activation="relu", name="dec_1")(encoded)
    x = BatchNormalization()(x)
    x = Dense(64, activation="relu", name="dec_2")(x)
    x = BatchNormalization()(x)
    decoded = Dense(input_dim, activation="linear", name="ae_output")(x)

    ae = Model(inp, decoded, name="FraudAutoencoder")
    ae.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )

    ae.summary()

    # Train
    t0 = time.time()
    history = ae.fit(
        X_normal, X_normal,
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH,
        validation_split=0.1,
        shuffle=True,
        callbacks=[
            EarlyStopping(
                monitor="val_loss", patience=7,
                restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=3, min_lr=1e-6, verbose=1
            ),
        ],
        verbose=1
    )
    elapsed = time.time() - t0

    # Save
    ae.save(MODEL_DIR / "paysim_ae_model_v2.keras")
    print(f"\n  ✅ Autoencoder trained in {elapsed:.1f}s")
    print(f"  ✅ Saved to {MODEL_DIR / 'paysim_ae_model_v2.keras'}")
    print(f"  Final train loss: {history.history['loss'][-1]:.6f}")
    print(f"  Final val loss:   {history.history['val_loss'][-1]:.6f}")

    return ae


# ============================================================
# STEP 5: COMPUTE RECONSTRUCTION ERRORS
# ============================================================
def compute_ae_errors(ae, X_train_s, X_val_s, X_test_s, y_train, y_val, y_test):
    print_step(5, "COMPUTE RECONSTRUCTION ERRORS",
               "High error = transaction doesn't look normal = potential fraud")

    def recon_error(model, X):
        recon = model.predict(X, batch_size=2048, verbose=0)
        mse = np.mean(np.square(X - recon), axis=1)
        return np.log1p(mse)  # Log-scale for better threshold separation

    err_train = recon_error(ae, X_train_s)
    err_val   = recon_error(ae, X_val_s)
    err_test  = recon_error(ae, X_test_s)

    # Print distribution
    for name, err, y in [
        ("Train", err_train, y_train),
        ("Val",   err_val,   y_val),
        ("Test",  err_test,  y_test)
    ]:
        legit_err = err[y.values == 0].mean()
        fraud_err = err[y.values == 1].mean()
        print(f"  {name}: Legit avg error = {legit_err:.4f} | "
              f"Fraud avg error = {fraud_err:.4f} | "
              f"Ratio = {fraud_err/legit_err:.2f}x")

    return err_train, err_val, err_test


# ============================================================
# STEP 6: DERIVE AE ANOMALY THRESHOLD
# ============================================================
def derive_ae_threshold(err_val, y_val):
    print_step(6, "DERIVE AE ANOMALY THRESHOLD",
               "Finding the reconstruction error cutoff that maximizes F2-score\n"
               "     on the validation set. F2 prioritizes recall (catching fraud).")

    # Sweep thresholds
    percentiles = np.arange(90, 99.9, 0.1)
    best_f2 = 0
    best_thresh = 0
    best_recall = 0
    best_precision = 0

    for pct in percentiles:
        thresh = np.percentile(err_val[y_val.values == 0], pct)
        preds = (err_val >= thresh).astype(int)

        tp = ((preds == 1) & (y_val.values == 1)).sum()
        fp = ((preds == 1) & (y_val.values == 0)).sum()
        fn = ((preds == 0) & (y_val.values == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        beta = 2
        f2 = ((1 + beta**2) * precision * recall /
              (beta**2 * precision + recall)) if (precision + recall) > 0 else 0

        if f2 > best_f2:
            best_f2 = f2
            best_thresh = thresh
            best_recall = recall
            best_precision = precision

    print(f"  Best Threshold: {best_thresh:.6f}")
    print(f"  Best F2:        {best_f2:.4f}")
    print(f"  Recall:         {best_recall:.4f}")
    print(f"  Precision:      {best_precision:.4f}")

    # Save
    np.save(MODEL_DIR / "paysim_ae_threshold.npy", np.array([best_thresh]))
    print(f"  ✅ Saved to {MODEL_DIR / 'paysim_ae_threshold.npy'}")

    return best_thresh


# ============================================================
# STEP 7: TRAIN ISOLATION FOREST
# ============================================================
def train_isolation_forest(X_train_s, y_train):
    print_step(7, "TRAIN ISOLATION FOREST",
               "Statistical outlier detector. Points 'easy to isolate'\n"
               "     with random tree splits → anomaly. Trained on all data\n"
               "     (it's unsupervised, doesn't use labels).")

    t0 = time.time()
    iforest = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        max_samples='auto',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )

    # Train on normal transactions only
    X_normal = X_train_s[y_train.values == 0]
    iforest.fit(X_normal)
    elapsed = time.time() - t0

    # Save
    joblib.dump(iforest, MODEL_DIR / "paysim_iforest_v2.pkl")
    print(f"  ✅ Isolation Forest trained in {elapsed:.1f}s")
    print(f"  ✅ Saved to {MODEL_DIR / 'paysim_iforest_v2.pkl'}")

    return iforest


# ============================================================
# STEP 8: EVALUATE ON TEST SET
# ============================================================
def evaluate_on_test(ae, iforest, X_test_s, y_test, ae_threshold, err_test):
    print_step(8, "EVALUATE ON HELD-OUT TEST SET",
               "Testing both anomaly detectors on data never seen during training")

    # AE predictions
    ae_preds = (err_test >= ae_threshold).astype(int)

    # IForest predictions (-1 = anomaly, 1 = normal)
    if_raw = iforest.predict(X_test_s)
    if_preds = (if_raw == -1).astype(int)  # Convert to 0/1

    # IForest scores (lower = more anomalous)
    if_scores = -iforest.score_samples(X_test_s)  # Negate so higher = more anomalous

    # Combined: OR-logic (either flags → fraud)
    combined_preds = ((ae_preds == 1) | (if_preds == 1)).astype(int)

    y_true = y_test.values

    # Print results
    for name, preds in [
        ("Autoencoder", ae_preds),
        ("Isolation Forest", if_preds),
        ("Combined (AE OR IF)", combined_preds)
    ]:
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        tn = ((preds == 0) & (y_true == 0)).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n  📊 {name}:")
        print(f"     Recall:    {recall:.4f}  ({tp}/{tp+fn} frauds caught)")
        print(f"     Precision: {precision:.4f}")
        print(f"     F1:        {f1:.4f}")
        print(f"     TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")

    # ROC-AUC for AE (using continuous error scores)
    try:
        ae_roc = roc_auc_score(y_true, err_test)
        ae_pr  = average_precision_score(y_true, err_test)
        print(f"\n  📊 AE Ranking Metrics:")
        print(f"     ROC-AUC: {ae_roc:.4f}")
        print(f"     PR-AUC:  {ae_pr:.4f}")
    except Exception as e:
        print(f"  ⚠️  Could not compute AE ranking: {e}")

    # ROC-AUC for IForest
    try:
        if_roc = roc_auc_score(y_true, if_scores)
        if_pr  = average_precision_score(y_true, if_scores)
        print(f"\n  📊 IForest Ranking Metrics:")
        print(f"     ROC-AUC: {if_roc:.4f}")
        print(f"     PR-AUC:  {if_pr:.4f}")
    except Exception as e:
        print(f"  ⚠️  Could not compute IForest ranking: {e}")

    return ae_preds, if_preds, combined_preds


# ============================================================
# MAIN
# ============================================================
def main():
    print_header("PAYSIM LAYER 2 — AUTOENCODER + ISOLATION FOREST RETRAINING")

    # Step 1
    df = load_data()

    # Step 2
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    del df; gc.collect()

    # Step 3
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    # Step 4
    ae = train_autoencoder(X_train_s, y_train)

    # Step 5
    err_train, err_val, err_test = compute_ae_errors(
        ae, X_train_s, X_val_s, X_test_s, y_train, y_val, y_test
    )

    # Step 6
    ae_threshold = derive_ae_threshold(err_val, y_val)

    # Step 7
    iforest = train_isolation_forest(X_train_s, y_train)

    # Step 8
    evaluate_on_test(ae, iforest, X_test_s, y_test, ae_threshold, err_test)

    # Summary
    print_header("LAYER 2 TRAINING COMPLETE!")
    print(f"  Files saved to: {MODEL_DIR}/")
    print(f"    • paysim_ae_model_v2.keras  (Autoencoder)")
    print(f"    • paysim_ae_threshold.npy   (Anomaly threshold)")
    print(f"    • paysim_ae_scaler.pkl      (Feature scaler)")
    print(f"    • paysim_iforest_v2.pkl     (Isolation Forest)")
    print(f"\n  Next: Run retrain_paysim_v2_lstm.py for Layer 3")


if __name__ == "__main__":
    main()
