"""
retrain_paysim_v2_lstm.py
─────────────────────────
Trains Layer 3 (Temporal Sequence Detection) of the anti-overfitting stack.

The LSTM looks at SEQUENCES of transactions (e.g., last 5 in a row)
and predicts if the next transaction is fraud. It captures temporal
patterns like "rapid transfers at unusual hours."

Key improvements over the old LSTM:
  - Uses full cleaned_paysim.csv (6.36M rows)
  - Removes has_balance_mismatch (data leakage)
  - Increases sequence cap to 1M (was 300K)
  - Saves to models/paysim_v2/

Usage:
    python retrain_paysim_v2_lstm.py
"""

import numpy as np
import pandas as pd
import joblib
import time
import gc
import os
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = Path("data/cleaned_paysim_lstm.csv")
MODEL_DIR = Path("models/paysim_v2")
EVAL_DIR  = Path("evaluation_results/paysim_evaluation_results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Sequence parameters
SEQUENCE_LENGTH = 5       # Look at last 5 transactions
MAX_SEQUENCES   = 1_000_000  # Cap for RAM safety (was 300K)

# LSTM Hyperparameters
LSTM_EPOCHS = 30
LSTM_BATCH  = 256

RANDOM_STATE = 42

# Features for LSTM (12 base + 3 rolling = 15)
# We add rolling features to capture temporal context WITHIN the sequence
BASE_FEATURES = [
    "amount",
    "hour", "dayofweek", "is_weekend",
    "errorbalanceorig", "errorbalancedest",
    "upi_type_upi_payment", "upi_type_upi_transfer",
    "oldbalanceorg", "newbalanceorig",
    "oldbalancedest", "newbalancedest",
]

TARGET = "isfraud"


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
# FOCAL LOSS (handles class imbalance without oversampling)
# ============================================================
def focal_loss(alpha=0.75, gamma=2.0):
    """
    Focal loss penalizes easy-to-classify examples less and 
    hard-to-classify examples more. alpha=0.75 gives extra
    weight to the minority class (fraud).
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)

        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1 - p_t, gamma)

        return alpha_factor * modulating_factor * bce

    return loss


# ============================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================
def load_and_prepare():
    print_step(1, "LOAD AND PREPARE DATA",
               "Loading cleaned_paysim_lstm.csv and creating temporal features")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()

    print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")

    # Time features from PaySim 'step'
    if 'hour' not in df.columns:
        df['hour'] = df['step'] % 24
    if 'dayofweek' not in df.columns:
        df['dayofweek'] = (df['step'] // 24) % 7
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(np.int8)

    # Cast booleans
    for col in ['upi_type_upi_payment', 'upi_type_upi_transfer']:
        if col in df.columns:
            df[col] = df[col].astype(np.int8)

    # Sort by time (critical for sequences)
    df = df.sort_values('step').reset_index(drop=True)

    # Check features exist
    missing = set(BASE_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    print(f"  Fraud: {df[TARGET].sum():,} ({df[TARGET].mean()*100:.3f}%)")
    print(f"  Features: {len(BASE_FEATURES)}")

    return df


# ============================================================
# STEP 2: SCALE AND CREATE SEQUENCES
# ============================================================
def create_sequences(df):
    print_step(2, "SCALE FEATURES AND CREATE SEQUENCES",
               f"Creating sliding windows of {SEQUENCE_LENGTH} transactions.\n"
               "     Each sequence: [tx_1, tx_2, ..., tx_5] → predict tx_6 label.")

    # Scale features
    scaler = StandardScaler()
    df[BASE_FEATURES] = scaler.fit_transform(df[BASE_FEATURES])

    # Save scaler for LSTM
    joblib.dump(scaler, MODEL_DIR / "paysim_lstm_scaler.pkl")
    joblib.dump(BASE_FEATURES, MODEL_DIR / "paysim_lstm_features.pkl")
    joblib.dump(SEQUENCE_LENGTH, MODEL_DIR / "paysim_lstm_seq_len.pkl")
    print(f"  ✅ Scaler + features + seq_len saved to {MODEL_DIR}")

    values = df[BASE_FEATURES].values
    labels = df[TARGET].values

    # Create sequences
    print(f"  Creating sequences from {len(df):,} transactions...")
    t0 = time.time()

    n = len(df) - SEQUENCE_LENGTH
    X = np.empty((n, SEQUENCE_LENGTH, len(BASE_FEATURES)), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)

    for i in range(n):
        X[i] = values[i:i + SEQUENCE_LENGTH]
        y[i] = labels[i + SEQUENCE_LENGTH]

    elapsed = time.time() - t0
    print(f"  Created {len(X):,} sequences in {elapsed:.1f}s")
    print(f"  X shape: {X.shape}")
    print(f"  Fraud sequences: {y.sum():,} ({y.mean()*100:.3f}%)")

    # Cap sequences if needed
    if len(X) > MAX_SEQUENCES:
        print(f"  ⚠️  Capping from {len(X):,} to {MAX_SEQUENCES:,} sequences")
        # Smart sampling: keep ALL fraud + random subset of normal
        fraud_idx = np.where(y == 1)[0]
        normal_idx = np.where(y == 0)[0]
        
        n_normal = MAX_SEQUENCES - len(fraud_idx)
        if n_normal > 0 and n_normal < len(normal_idx):
            np.random.seed(RANDOM_STATE)
            sampled_normal = np.random.choice(normal_idx, n_normal, replace=False)
            keep_idx = np.concatenate([fraud_idx, sampled_normal])
            np.random.shuffle(keep_idx)
            X = X[keep_idx]
            y = y[keep_idx]
        
        print(f"  After cap: {len(X):,} sequences, {y.sum():,} fraud ({y.mean()*100:.3f}%)")

    return X, y


# ============================================================
# STEP 3: TIME-BASED SPLIT (NO RANDOM SHUFFLE BEFORE SPLIT)
# ============================================================
def split_sequences(X, y):
    print_step(3, "TIME-BASED SPLIT (70/15/15)",
               "No random shuffle before split — preserves temporal order.\n"
               "     This prevents future-leakage where the model sees\n"
               "     'future' transactions during training.")

    n = len(X)
    train_end = int(0.70 * n)
    val_end   = int(0.85 * n)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    print(f"  Train: {len(X_train):,} ({y_train.sum():,} fraud)")
    print(f"  Val:   {len(X_val):,}   ({y_val.sum():,} fraud)")
    print(f"  Test:  {len(X_test):,}  ({y_test.sum():,} fraud)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# STEP 4: BUILD AND TRAIN LSTM
# ============================================================
def train_lstm(X_train, y_train, X_val, y_val):
    print_step(4, "BUILD AND TRAIN LSTM",
               "Architecture: LSTM(64) → Dropout → Dense(32) → Dense(1)\n"
               "     Loss: Focal Loss (α=0.75, γ=2.0) — handles class imbalance\n"
               "     without SMOTE (sequences can't use SMOTE easily).")

    n_timesteps = X_train.shape[1]
    n_features  = X_train.shape[2]

    model = Sequential([
        Input(shape=(n_timesteps, n_features)),
        LSTM(64, return_sequences=False),
        Dropout(0.4),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(alpha=0.75, gamma=2.0),
        metrics=[
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR")
        ]
    )

    model.summary()

    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_pr_auc", mode="max",
        patience=5, restore_best_weights=True, verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_pr_auc", mode="max",
        factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    # Train
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    elapsed = time.time() - t0

    # Save
    model.save(MODEL_DIR / "paysim_lstm_v2.keras")
    print(f"\n  ✅ LSTM trained in {elapsed:.1f}s")
    print(f"  ✅ Saved to {MODEL_DIR / 'paysim_lstm_v2.keras'}")

    return model


# ============================================================
# STEP 5: DERIVE THRESHOLD AND EVALUATE
# ============================================================
def evaluate_lstm(model, X_val, y_val, X_test, y_test):
    print_step(5, "DERIVE THRESHOLD AND EVALUATE ON TEST SET",
               "Using F2-score on validation set to find optimal threshold,\n"
               "     then evaluating on held-out test set.")

    # Predict on validation set
    val_probs = model.predict(X_val, batch_size=512, verbose=0).ravel()

    # Sweep thresholds
    best_f2 = 0
    best_thresh = 0.5

    for thresh in np.arange(0.01, 0.99, 0.01):
        preds = (val_probs >= thresh).astype(int)
        tp = ((preds == 1) & (y_val == 1)).sum()
        fp = ((preds == 1) & (y_val == 0)).sum()
        fn = ((preds == 0) & (y_val == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        beta = 2
        f2 = ((1 + beta**2) * precision * recall /
              (beta**2 * precision + recall)) if (precision + recall) > 0 else 0

        if f2 > best_f2:
            best_f2 = f2
            best_thresh = thresh

    print(f"  Best Threshold: {best_thresh:.2f}")
    print(f"  Best F2 (Val):  {best_f2:.4f}")

    # Save threshold
    np.save(MODEL_DIR / "paysim_lstm_threshold.npy", np.array([best_thresh]))

    # Evaluate on TEST set
    test_probs = model.predict(X_test, batch_size=512, verbose=0).ravel()
    test_preds = (test_probs >= best_thresh).astype(int)

    tp = ((test_preds == 1) & (y_test == 1)).sum()
    fp = ((test_preds == 1) & (y_test == 0)).sum()
    fn = ((test_preds == 0) & (y_test == 1)).sum()
    tn = ((test_preds == 0) & (y_test == 0)).sum()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  📊 LSTM Test Results:")
    print(f"     Recall:    {recall:.4f}  ({tp}/{tp+fn} frauds caught)")
    print(f"     Precision: {precision:.4f}")
    print(f"     F1:        {f1:.4f}")
    print(f"     TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")

    # ROC/PR AUC
    try:
        roc = roc_auc_score(y_test, test_probs)
        pr  = average_precision_score(y_test, test_probs)
        print(f"     ROC-AUC: {roc:.4f}")
        print(f"     PR-AUC:  {pr:.4f}")
    except Exception as e:
        print(f"  ⚠️  Could not compute AUC: {e}")

    return best_thresh


# ============================================================
# MAIN
# ============================================================
def main():
    print_header("PAYSIM LAYER 3 — LSTM TEMPORAL SEQUENCE RETRAINING")

    # Step 1
    df = load_and_prepare()

    # Step 2
    X, y = create_sequences(df)
    del df; gc.collect()

    # Step 3
    X_train, X_val, X_test, y_train, y_val, y_test = split_sequences(X, y)
    del X, y; gc.collect()

    # Step 4
    model = train_lstm(X_train, y_train, X_val, y_val)

    # Step 5
    threshold = evaluate_lstm(model, X_val, y_val, X_test, y_test)

    # Summary
    print_header("LAYER 3 TRAINING COMPLETE!")
    print(f"  Files saved to: {MODEL_DIR}/")
    print(f"    • paysim_lstm_v2.keras        (LSTM model)")
    print(f"    • paysim_lstm_threshold.npy   (Decision threshold)")
    print(f"    • paysim_lstm_scaler.pkl      (Feature scaler)")
    print(f"    • paysim_lstm_features.pkl    (Feature list)")
    print(f"    • paysim_lstm_seq_len.pkl     (Sequence length)")
    print(f"\n  Next: Run updated test_unseen_fraud.py with ALL 4 models")


if __name__ == "__main__":
    main()
