"""
train_paysim_v4_hybrid.py — V4 Experimental Pipeline
══════════════════════════════════════════════════════
Two-path hybrid fraud detection with LSTM/GRU sequential model integration.

What's new in V4 vs V3:
  1. BiLSTM with Attention trained on SMOTE-balanced sequences (18 features)
  2. BiGRU with Attention trained as a comparison model
  3. Winner's sequential score becomes the 20th input feature for XGBoost/RF
  4. Path B enhanced: AE + IForest + Sequential anomaly detection

Protection:
  - All artifacts saved to models/paysim_v4_experiment/ (V3 is NEVER touched)
  - If V4 underperforms, simply delete that folder

Usage:
  python src/train_paysim_v4_hybrid.py
  python src/train_paysim_v4_hybrid.py --data-pct 0.10   # Quick test on 10% data
"""

import numpy as np
import pandas as pd
import joblib
import os
import gc
import json
import time
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    f1_score,
)
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, GRU, Bidirectional,
    Dropout, BatchNormalization, Layer,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Suppress TF info logs
tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ══════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="Train V4 Hybrid Ensemble")
parser.add_argument(
    "--data-pct", type=float, default=1.0,
    help="Fraction of data to use (0.05-1.0). Use 0.05-0.10 for quick testing."
)
args = parser.parse_args()

# ══════════════════════════════════════════════════════
# PHASE 0: CONFIGURATION
# ══════════════════════════════════════════════════════
"""
All configuration in one place. Easy to tune without digging through code.
"""
DATA_PATH = "data/cleaned_paysim_lstm.csv"
V4_MODEL_DIR = "models/paysim_v4_experiment"

# Autoencoder config (same architecture as V3 — proven to work)
AE_EPOCHS = 30
AE_BATCH = 1024
AE_PATIENCE = 5

# Sequential model config
SEQ_LENGTH = 5           # Window of 5 consecutive transactions
SEQ_EPOCHS = 30          # Max epochs for BiLSTM/GRU
SEQ_BATCH = 256          # Batch size
SEQ_PATIENCE = 7         # Early stopping patience

# SMOTE config
SMOTE_TARGET_RATIO = 0.10  # Upsample fraud to 10% of training data
MAX_SMOTE_SAMPLES = 500_000  # RAM safety cap

# Supervised model config
XGB_ESTIMATORS = 400
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05
RF_ESTIMATORS = 300
RF_MAX_DEPTH = 12

# Threshold config
TARGET_RECALL = 0.80     # Business constraint for Path A

# Isolation Forest config
IFOREST_ESTIMATORS = 200
IFOREST_CONTAMINATION = 0.001

# Data fraction (from args)
DATA_PCT = args.data_pct

os.makedirs(V4_MODEL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════
# CUSTOM ATTENTION LAYER
# ══════════════════════════════════════════════════════
"""
Bahdanau (Additive) Attention Mechanism
────────────────────────────────────────
WHY: Standard LSTM treats all timesteps equally when producing its final
hidden state. But in fraud detection, the LAST transaction in a sequence
is usually the suspicious one — we want the model to PAY MORE ATTENTION
to the most relevant timesteps.

HOW it works:
  1. Each hidden state h_t gets projected through a learned weight matrix W
  2. A tanh activation produces alignment scores
  3. Softmax normalizes scores → attention weights (α_1, α_2, ..., α_T)
  4. Weighted sum: context = Σ(α_t * h_t) → single vector capturing the
     most relevant information across all timesteps

ACADEMIC REFERENCE:
  Bahdanau et al. (2015) — "Neural Machine Translation by Jointly
  Learning to Align and Translate"
"""

class BahdanauAttention(Layer):
    """
    Additive attention mechanism for temporal sequences.

    Input:  (batch_size, timesteps, features) — sequence of hidden states
    Output: (batch_size, features) — context vector (weighted sum of states)
    """

    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        # W projects hidden states into a learned representation space
        self.W = Dense(units, use_bias=True, name="attention_projection")
        # V maps projected states to scalar alignment scores
        self.V = Dense(1, use_bias=False, name="attention_score")

    def call(self, hidden_states):
        """
        hidden_states: (batch, timesteps, features)
        Returns:       (batch, features) — the context vector
        """
        # Step 1: Project each hidden state → (batch, timesteps, units)
        projected = tf.keras.activations.tanh(self.W(hidden_states))

        # Step 2: Compute scalar alignment score per timestep → (batch, timesteps, 1)
        score = self.V(projected)

        # Step 3: Softmax over timesteps → attention weights that sum to 1
        # α_t = exp(score_t) / Σ exp(score_t)
        attention_weights = tf.keras.activations.softmax(score, axis=1)

        # Step 4: Weighted sum of hidden states → context vector
        # context = Σ(α_t * h_t)
        context_vector = tf.reduce_sum(hidden_states * attention_weights, axis=1)

        return context_vector

    def get_config(self):
        """Required for model serialization (saving/loading)."""
        config = super().get_config()
        config.update({"units": self.units})
        return config


# ══════════════════════════════════════════════════════
# FOCAL LOSS (SAME AS V3 LSTM — PROVEN FOR IMBALANCED DATA)
# ══════════════════════════════════════════════════════
"""
Focal Loss — addresses class imbalance by down-weighting easy examples.

Standard cross-entropy gives equal weight to all samples. With 0.13% fraud,
the model can achieve 99.87% accuracy by predicting everything as legit.
Focal loss fixes this by:
  - α (alpha): Weights the fraud class more heavily (α=0.75 means fraud
    gets 3x the weight of legitimate)
  - γ (gamma): Focuses on HARD examples. Easy examples (high p_t) get their
    loss reduced by (1-p_t)^γ, so the model concentrates on misclassified cases.

REFERENCE: Lin et al. (2017) — "Focal Loss for Dense Object Detection"
"""

def focal_loss(alpha=0.75, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        # Numerical stability: clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Standard binary cross-entropy
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)

        # p_t = probability of correct class
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

        # α factor: weight for positive class
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

        # Modulating factor: (1 - p_t)^γ — reduces loss for easy examples
        modulating_factor = tf.pow(1 - p_t, gamma)

        return alpha_factor * modulating_factor * bce

    return loss


# ══════════════════════════════════════════════════════
# PHASE 1: DATA LOADING & TIME-AWARE SPLITTING
# ══════════════════════════════════════════════════════
print("=" * 70)
print(" V4 HYBRID PIPELINE — TRAINING")
print("=" * 70)
print(f"\nData fraction: {DATA_PCT:.0%}")

print("\n📂 Phase 1: Loading PaySim dataset...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower().str.strip()

# Sort by time (critical — prevents future data leaking into past)
df = df.sort_values("step").reset_index(drop=True)

# Sample data if requested (for quick testing)
if DATA_PCT < 1.0:
    n_sample = int(len(df) * DATA_PCT)
    df = df.iloc[:n_sample].reset_index(drop=True)
    print(f"   Using {len(df):,} transactions ({DATA_PCT:.0%} of full dataset)")

# ── Engineer velocity features (same as V3) ──
"""
These 6 behavioral features capture PER-ACCOUNT patterns across time.
They are critical for detecting novel fraud because they reveal:
  - How active the account is (tx_count, amount_cumul)
  - Whether the current transaction is unusual FOR THIS ACCOUNT (amt_vs_avg)
  - How quickly the account is transacting (time_since_last)
  - How aggressively the balance is being drained (amt_to_bal_ratio, balance_velocity)

Computed using vectorized pandas groupby — no Python lambdas (~30s on 6.3M rows).
"""
print("\n⚙️  Phase 1b: Engineering velocity features...")
import time as _time
_t0 = _time.time()

# Ensure time-based features exist
if "hour" not in df.columns:
    df["hour"] = df["step"] % 24
if "dayofweek" not in df.columns:
    df["dayofweek"] = (df["step"] // 24) % 7
if "is_weekend" not in df.columns:
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)

for col in ["upi_type_upi_payment", "upi_type_upi_transfer"]:
    if col in df.columns:
        df[col] = df[col].astype(np.int8)

# Feature 1: tx_count_cumul — How many transactions has this account made so far?
print("   Computing tx_count_cumul...")
df["tx_count_cumul"] = df.groupby("nameorig").cumcount() + 1

# Feature 2: amount_cumul — Total amount moved by this account so far
print("   Computing amount_cumul...")
df["amount_cumul"] = df.groupby("nameorig")["amount"].cumsum()

# Feature 3: amt_vs_avg — Is this transaction unusually large for this account?
print("   Computing amt_vs_avg...")
df["amt_vs_avg"] = df["amount"] / (df["amount_cumul"] / df["tx_count_cumul"] + 1e-6)

# Feature 4: time_since_last — How long since this account's last transaction?
print("   Computing time_since_last...")
df["time_since_last"] = df.groupby("nameorig")["step"].diff().fillna(48)
df["time_since_last"] = df["time_since_last"].clip(0, 96)

# Feature 5: amt_to_bal_ratio — What fraction of balance is being moved?
print("   Computing amt_to_bal_ratio...")
df["amt_to_bal_ratio"] = df["amount"] / (df["oldbalanceorg"] + 1e-6)

# Feature 6: balance_velocity — Rate of balance drain (negative = draining)
print("   Computing balance_velocity...")
df["balance_velocity"] = (
    (df["newbalanceorig"] - df["oldbalanceorg"]) /
    (df["amount"] + 1e-6)
)

# Log-transform heavy-tailed features (same as V3)
df["amount_cumul"] = np.log1p(df["amount_cumul"].clip(0))
df["tx_count_cumul"] = np.log1p(df["tx_count_cumul"])
df["amt_to_bal_ratio"] = np.log1p(df["amt_to_bal_ratio"].clip(0))

print(f"   ✅ Velocity features engineered in {_time.time() - _t0:.1f}s")

# ── Define features (18 base features — same as V3) ──
TARGET = "isfraud"

# Remove data-leakage feature if present
if "has_balance_mismatch" in df.columns:
    print("   ⚠️  Dropped 'has_balance_mismatch' (data leakage)")

FEATURES = [
    # Original 12 base features
    "amount", "oldbalanceorg", "newbalanceorig",
    "oldbalancedest", "newbalancedest",
    "hour", "dayofweek", "is_weekend",
    "errorbalanceorig", "errorbalancedest",
    "upi_type_upi_payment", "upi_type_upi_transfer",
    # 6 velocity features (same as V3)
    "tx_count_cumul", "amount_cumul", "amt_vs_avg",
    "time_since_last", "amt_to_bal_ratio", "balance_velocity",
]

# Verify all features exist
missing = set(FEATURES) - set(df.columns)
if missing:
    raise ValueError(f"Missing features after engineering: {missing}")

# Clean NaN/inf values
df[FEATURES] = df[FEATURES].fillna(0)
for col in FEATURES:
    df[col] = df[col].replace([np.inf, -np.inf], 0)

X = df[FEATURES].values.astype(np.float64)
y = df[TARGET].values.astype(np.int32)

print(f"   Total transactions: {len(df):,}")
print(f"   Features: {len(FEATURES)} → {FEATURES}")
print(f"   Fraud ratio: {y.mean():.4%} ({y.sum():,} frauds)")

# ── Time-aware split (70/15/15) ──
"""
WHY time-aware split (not random):
  - Random split would leak future information into training data
  - In production, the model only sees PAST transactions when predicting
  - Time-aware split simulates this: train on oldest 70%, test on newest 15%
"""
n = len(df)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print(f"\n   Split sizes:")
print(f"   Train: {X_train.shape[0]:,} ({y_train.mean():.4%} fraud)")
print(f"   Val:   {X_val.shape[0]:,} ({y_val.mean():.4%} fraud)")
print(f"   Test:  {X_test.shape[0]:,} ({y_test.mean():.4%} fraud)")


# ══════════════════════════════════════════════════════
# PHASE 2: FEATURE SCALING
# ══════════════════════════════════════════════════════
"""
StandardScaler: transforms each feature to mean=0, std=1.

WHY: Neural networks (AE, LSTM) are sensitive to feature scale.
'amount' can range 0-10M while 'is_weekend' is 0/1. Without scaling,
the large-range features would dominate gradient updates.

IMPORTANT: Fit scaler on TRAINING DATA ONLY, then transform val/test.
Fitting on all data would leak val/test distribution into training.
"""
print("\n⚙️  Phase 2: Scaling features...")

scaler_base = StandardScaler()
X_train_scaled = scaler_base.fit_transform(X_train)
X_val_scaled = scaler_base.transform(X_val)
X_test_scaled = scaler_base.transform(X_test)

# Save base scaler (18 features) — will extend to 20 features later
joblib.dump(scaler_base, f"{V4_MODEL_DIR}/paysim_v4_base_scaler.pkl")
joblib.dump(FEATURES, f"{V4_MODEL_DIR}/paysim_v4_features.pkl")

print(f"   Scaler fit on {X_train_scaled.shape[0]:,} training samples")


# ══════════════════════════════════════════════════════
# PHASE 3: AUTOENCODER TRAINING (SAME AS V3)
# ══════════════════════════════════════════════════════
"""
Autoencoder (AE) — learns to RECONSTRUCT normal transactions.

Architecture: Input(18) → 64 → 32 → 16 (bottleneck) → 32 → 64 → 18
Loss: MSE (Mean Squared Error)
Trained on: LEGITIMATE transactions ONLY

WHY train on normal only:
  - AE learns the "shape" of legitimate transactions
  - When a fraudulent transaction is fed in, the AE can't reconstruct it well
  - High reconstruction error = suspicious transaction

The reconstruction error becomes the 19th feature for XGBoost/RF.
"""
print("\n🔧 Phase 3: Training Autoencoder...")

# Only train on legitimate (non-fraud) transactions
X_train_normal = X_train_scaled[y_train == 0]
print(f"   Training AE on {X_train_normal.shape[0]:,} legitimate transactions")

input_dim = X_train_normal.shape[1]

# Build AE architecture (same as V3 — proven to work)
ae_input = Input(shape=(input_dim,), name="ae_input")
x = Dense(64, activation="relu", name="encoder_1")(ae_input)
x = Dense(32, activation="relu", name="encoder_2")(x)
encoded = Dense(16, activation="relu", name="bottleneck")(x)
x = Dense(32, activation="relu", name="decoder_1")(encoded)
x = Dense(64, activation="relu", name="decoder_2")(x)
ae_output = Dense(input_dim, activation="linear", name="ae_output")(x)

ae_model = Model(ae_input, ae_output, name="autoencoder")
ae_model.compile(optimizer="adam", loss="mse")

print(f"   AE parameters: {ae_model.count_params():,}")

ae_history = ae_model.fit(
    X_train_normal, X_train_normal,
    epochs=AE_EPOCHS,
    batch_size=AE_BATCH,
    validation_split=0.1,
    shuffle=True,
    callbacks=[EarlyStopping(patience=AE_PATIENCE, restore_best_weights=True)],
    verbose=1,
)

ae_model.save(f"{V4_MODEL_DIR}/paysim_v4_ae.keras")
print(f"   ✅ AE saved. Final val_loss: {min(ae_history.history['val_loss']):.6f}")


# ══════════════════════════════════════════════════════
# PHASE 3b: COMPUTE AE RECONSTRUCTION ERROR
# ══════════════════════════════════════════════════════
"""
Reconstruction error = how "different" the AE's output is from the input.

We use log1p(MSE) instead of raw MSE because:
  - Raw MSE values can be very small (1e-5 to 1e-3 for legitimate)
  - log1p spreads them out, making it easier for XGBoost to find splits
  - log1p(x) = log(1 + x), so 0 maps to 0 (nice property)
"""

def compute_ae_error(model, X_scaled, batch_size=2048):
    """Compute log-scaled AE reconstruction error per sample."""
    recon = model.predict(X_scaled, batch_size=batch_size, verbose=0)
    mse_per_sample = np.mean(np.square(X_scaled - recon), axis=1)
    return np.log1p(mse_per_sample)


print("\n   Computing AE reconstruction errors...")
ae_err_train = compute_ae_error(ae_model, X_train_scaled)
ae_err_val = compute_ae_error(ae_model, X_val_scaled)
ae_err_test = compute_ae_error(ae_model, X_test_scaled)

print(f"   AE error stats (train):")
print(f"     Legitimate mean: {ae_err_train[y_train == 0].mean():.6f}")
print(f"     Fraud mean:      {ae_err_train[y_train == 1].mean():.6f}")
print(f"     Separation ratio: {ae_err_train[y_train == 1].mean() / (ae_err_train[y_train == 0].mean() + 1e-10):.1f}x")

# Compute AE anomaly threshold (95th percentile of legitimate errors)
ae_threshold = np.percentile(ae_err_train[y_train == 0], 95)
np.save(f"{V4_MODEL_DIR}/paysim_v4_ae_threshold.npy", np.array([ae_threshold]))
print(f"   AE anomaly threshold (95th pct): {ae_threshold:.6f}")


# ══════════════════════════════════════════════════════
# PHASE 4: LSTM SEQUENCE CREATION + SMOTE BALANCING
# ══════════════════════════════════════════════════════
"""
Creating temporal sequences for the LSTM/GRU models.

A sequence = window of SEQ_LENGTH consecutive transactions.
For each sequence, the LABEL is the fraud status of the LAST transaction.

Example (SEQ_LENGTH=5):
  Sequence: [txn_96, txn_97, txn_98, txn_99, txn_100]
  Label:    isFraud(txn_100)

WHY sequences: LSTM excels at spotting PATTERNS over time.
Example pattern: "3 normal transactions → sudden large amount → different
destination" = suspicious sequential behavior that single-transaction
models can't capture.

SMOTE ON SEQUENCES:
  - Raw data has 0.13% fraud sequences — way too few to learn from
  - SMOTE creates synthetic fraud sequences by interpolating between
    similar real fraud sequences
  - We flatten (samples, 5, 18) → (samples, 90), apply SMOTE, reshape back
  - Target: 10% fraud ratio (same as V3 XGBoost training)
"""
print("\n🔗 Phase 4: Creating LSTM sequences + SMOTE balancing...")


def create_sequences(X, y, seq_length):
    """
    Create sliding-window sequences from time-ordered data.

    Input:
      X: (n_samples, n_features) — feature matrix, time-ordered
      y: (n_samples,) — labels
      seq_length: int — window size

    Output:
      X_seq: (n_sequences, seq_length, n_features)
      y_seq: (n_sequences,) — label of the LAST element in each window
    """
    sequences = []
    labels = []

    for i in range(seq_length, len(X)):
        # Window of `seq_length` consecutive transactions
        sequences.append(X[i - seq_length : i])
        # Label is the fraud status of the current (last) transaction
        labels.append(y[i])

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int32)


# Create sequences from SCALED features (LSTM needs scaled input)
print(f"   Creating sequences (window={SEQ_LENGTH})...")
X_seq_train, y_seq_train = create_sequences(X_train_scaled, y_train, SEQ_LENGTH)
X_seq_val, y_seq_val = create_sequences(X_val_scaled, y_val, SEQ_LENGTH)
X_seq_test, y_seq_test = create_sequences(X_test_scaled, y_test, SEQ_LENGTH)

print(f"   Sequence shapes:")
print(f"     Train: {X_seq_train.shape} ({y_seq_train.mean():.4%} fraud)")
print(f"     Val:   {X_seq_val.shape}")
print(f"     Test:  {X_seq_test.shape}")

# ── Apply SMOTE to training sequences ──
"""
SMOTE (Synthetic Minority Oversampling Technique):
  1. For each fraud sequence, find its k nearest fraud neighbors
  2. Create synthetic fraud sequences by interpolating between them
  3. Result: more fraud examples for the model to learn from

We flatten 3D → 2D because SMOTE works on 2D arrays:
  (n_samples, 5, 18) → (n_samples, 90)
  Apply SMOTE on the flattened vectors
  (n_samples_new, 90) → (n_samples_new, 5, 18)

The interpolation preserves temporal structure because:
  - Similar fraud sequences have similar temporal patterns
  - Interpolating between them creates plausible fraud sequences
"""
print(f"\n   Applying SMOTE (target fraud ratio: {SMOTE_TARGET_RATIO:.0%})...")

from imblearn.over_sampling import SMOTE

n_train_seq = X_seq_train.shape[0]
n_features_flat = SEQ_LENGTH * X_seq_train.shape[2]

# Flatten: (samples, 5, 18) → (samples, 90)
X_flat = X_seq_train.reshape(n_train_seq, n_features_flat)

# Cap samples for RAM safety
if len(X_flat) > MAX_SMOTE_SAMPLES:
    print(f"   ⚠️  Capping at {MAX_SMOTE_SAMPLES:,} samples for RAM safety")
    idx = np.random.choice(len(X_flat), MAX_SMOTE_SAMPLES, replace=False)
    X_flat = X_flat[idx]
    y_seq_train_smote = y_seq_train[idx]
else:
    y_seq_train_smote = y_seq_train

# Calculate how many minority samples we need
n_majority = (y_seq_train_smote == 0).sum()
n_minority_target = int(n_majority * SMOTE_TARGET_RATIO / (1 - SMOTE_TARGET_RATIO))

print(f"   Before SMOTE: {(y_seq_train_smote == 1).sum():,} fraud / {(y_seq_train_smote == 0).sum():,} legit")

smote = SMOTE(
    sampling_strategy={1: max(n_minority_target, (y_seq_train_smote == 1).sum())},
    k_neighbors=min(5, (y_seq_train_smote == 1).sum() - 1),
    random_state=42,
)

X_flat_balanced, y_seq_balanced = smote.fit_resample(X_flat, y_seq_train_smote)

# Reshape back: (samples_new, 90) → (samples_new, 5, 18)
X_seq_train_balanced = X_flat_balanced.reshape(-1, SEQ_LENGTH, X_seq_train.shape[2])
y_seq_train_balanced = y_seq_balanced

print(f"   After SMOTE:  {(y_seq_train_balanced == 1).sum():,} fraud / {(y_seq_train_balanced == 0).sum():,} legit")
print(f"   New fraud ratio: {y_seq_train_balanced.mean():.2%}")

# Shuffle the balanced dataset
shuffle_idx = np.random.permutation(len(X_seq_train_balanced))
X_seq_train_balanced = X_seq_train_balanced[shuffle_idx]
y_seq_train_balanced = y_seq_train_balanced[shuffle_idx]

# Free memory
del X_flat, X_flat_balanced
gc.collect()


# ══════════════════════════════════════════════════════
# PHASE 5: BiLSTM WITH ATTENTION
# ══════════════════════════════════════════════════════
"""
Bidirectional LSTM with Bahdanau Attention
──────────────────────────────────────────

Architecture:
  Input(5, 18)                          ← 5 timesteps × 18 features
  → Bidirectional(LSTM(64, return_sequences=True))  ← reads forward + backward
  → BahdanauAttention(32)               ← learns which timesteps matter
  → BatchNormalization()                 ← stabilizes training
  → Dense(32, relu) → Dropout(0.3)      ← classification head
  → Dense(1, sigmoid)                   ← fraud probability

WHY Bidirectional:
  - Standard LSTM reads only left-to-right (past → present)
  - BiLSTM also reads right-to-left (present → past)
  - This lets it see: "this transaction is suspicious BECAUSE OF
    what the sequence looks like from both directions"
  - Output: 128-dim hidden state per timestep (64 forward + 64 backward)

WHY Attention:
  - Without attention, LSTM compresses the entire sequence into one
    fixed-size vector (the last hidden state). Information from early
    timesteps gets "forgotten"
  - Attention learns to WEIGHT each timestep differently
  - For fraud: the suspicious transaction (usually last) gets high weight
  - We can inspect attention weights for EXPLAINABILITY
"""
print("\n🧠 Phase 5: Training BiLSTM with Attention...")

seq_input_shape = (SEQ_LENGTH, X_seq_train.shape[2])  # (5, 18)

# Build BiLSTM + Attention model
bilstm_input = Input(shape=seq_input_shape, name="bilstm_input")

# Bidirectional LSTM: outputs (batch, 5, 128) — 64 forward + 64 backward per timestep
x = Bidirectional(
    LSTM(64, return_sequences=True, name="lstm_forward"),
    name="bidirectional_lstm",
)(bilstm_input)

# Attention: (batch, 5, 128) → (batch, 128) — weighted sum across timesteps
x = BahdanauAttention(units=32, name="attention")(x)

# Classification head
x = BatchNormalization(name="bn")(x)
x = Dense(32, activation="relu", name="dense_1")(x)
x = Dropout(0.3, name="dropout")(x)
bilstm_output = Dense(1, activation="sigmoid", name="fraud_prob")(x)

bilstm_model = Model(bilstm_input, bilstm_output, name="BiLSTM_Attention")
bilstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=focal_loss(alpha=0.75, gamma=2.0),
    metrics=[
        tf.keras.metrics.AUC(name="roc_auc"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
    ],
)

bilstm_model.summary()
print(f"\n   BiLSTM parameters: {bilstm_model.count_params():,}")

# Train BiLSTM
bilstm_callbacks = [
    EarlyStopping(
        monitor="val_pr_auc", mode="max",
        patience=SEQ_PATIENCE, restore_best_weights=True,
    ),
    ReduceLROnPlateau(
        monitor="val_pr_auc", mode="max",
        factor=0.5, patience=3, min_lr=1e-5,
    ),
]

t0 = time.time()
bilstm_history = bilstm_model.fit(
    X_seq_train_balanced, y_seq_train_balanced,
    validation_data=(X_seq_val, y_seq_val),
    epochs=SEQ_EPOCHS,
    batch_size=SEQ_BATCH,
    callbacks=bilstm_callbacks,
    verbose=1,
)
bilstm_train_time = time.time() - t0

# Evaluate BiLSTM on test set
bilstm_pred_test = bilstm_model.predict(X_seq_test, batch_size=512).ravel()
bilstm_roc_auc = roc_auc_score(y_seq_test, bilstm_pred_test)
bilstm_pr_auc = average_precision_score(y_seq_test, bilstm_pred_test)

print(f"\n   ✅ BiLSTM Results:")
print(f"      ROC-AUC:      {bilstm_roc_auc:.4f}")
print(f"      PR-AUC:       {bilstm_pr_auc:.4f}")
print(f"      Training time: {bilstm_train_time:.1f}s")


# ══════════════════════════════════════════════════════
# PHASE 6: BiGRU WITH ATTENTION (COMPARISON)
# ══════════════════════════════════════════════════════
"""
Bidirectional GRU with Bahdanau Attention
─────────────────────────────────────────

Same architecture as BiLSTM but uses GRU cells instead.

GRU vs LSTM — Key Differences:
  ┌─────────────────┬──────────────────┬──────────────────┐
  │ Aspect          │ LSTM             │ GRU              │
  ├─────────────────┼──────────────────┼──────────────────┤
  │ Gates           │ 3 (input, forget,│ 2 (reset, update)│
  │                 │   output)        │                  │
  │ Parameters      │ More (~66K)      │ Fewer (~50K)     │
  │ Memory cell     │ Separate c_t     │ Combined with h_t│
  │ Training speed  │ Slower           │ ~25% faster      │
  │ Long sequences  │ Better           │ Slightly worse   │
  │ Short sequences │ Good             │ Often matches    │
  │ Overfitting risk│ Higher           │ Lower            │
  └─────────────────┴──────────────────┴──────────────────┘

WHY compare:
  Our sequences are SHORT (length=5). GRU's simpler architecture may
  generalize better with fewer parameters on short sequences.
  We train both and pick the winner objectively.
"""
print("\n🧠 Phase 6: Training BiGRU with Attention...")

# Build BiGRU + Attention model (mirrors BiLSTM architecture)
bigru_input = Input(shape=seq_input_shape, name="bigru_input")

x = Bidirectional(
    GRU(64, return_sequences=True, name="gru_forward"),
    name="bidirectional_gru",
)(bigru_input)

x = BahdanauAttention(units=32, name="gru_attention")(x)

x = BatchNormalization(name="gru_bn")(x)
x = Dense(32, activation="relu", name="gru_dense_1")(x)
x = Dropout(0.3, name="gru_dropout")(x)
bigru_output = Dense(1, activation="sigmoid", name="gru_fraud_prob")(x)

bigru_model = Model(bigru_input, bigru_output, name="BiGRU_Attention")
bigru_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=focal_loss(alpha=0.75, gamma=2.0),
    metrics=[
        tf.keras.metrics.AUC(name="roc_auc"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
    ],
)

bigru_model.summary()
print(f"\n   BiGRU parameters: {bigru_model.count_params():,}")

# Train BiGRU
bigru_callbacks = [
    EarlyStopping(
        monitor="val_pr_auc", mode="max",
        patience=SEQ_PATIENCE, restore_best_weights=True,
    ),
    ReduceLROnPlateau(
        monitor="val_pr_auc", mode="max",
        factor=0.5, patience=3, min_lr=1e-5,
    ),
]

t0 = time.time()
bigru_history = bigru_model.fit(
    X_seq_train_balanced, y_seq_train_balanced,
    validation_data=(X_seq_val, y_seq_val),
    epochs=SEQ_EPOCHS,
    batch_size=SEQ_BATCH,
    callbacks=bigru_callbacks,
    verbose=1,
)
bigru_train_time = time.time() - t0

# Evaluate BiGRU on test set
bigru_pred_test = bigru_model.predict(X_seq_test, batch_size=512).ravel()
bigru_roc_auc = roc_auc_score(y_seq_test, bigru_pred_test)
bigru_pr_auc = average_precision_score(y_seq_test, bigru_pred_test)

print(f"\n   ✅ BiGRU Results:")
print(f"      ROC-AUC:      {bigru_roc_auc:.4f}")
print(f"      PR-AUC:       {bigru_pr_auc:.4f}")
print(f"      Training time: {bigru_train_time:.1f}s")


# ══════════════════════════════════════════════════════
# PHASE 7: MODEL COMPARISON & WINNER SELECTION
# ══════════════════════════════════════════════════════
"""
Select the best sequential model based on PR-AUC.

WHY PR-AUC (not ROC-AUC):
  - With 99.87% legitimate transactions, ROC-AUC can be misleadingly high
  - PR-AUC is more informative for imbalanced datasets because it
    focuses on the MINORITY class (fraud) performance
  - A model with high PR-AUC is good at both FINDING fraud (recall)
    and NOT false-alarming on legitimate transactions (precision)
"""
print("\n" + "=" * 70)
print(" SEQUENTIAL MODEL COMPARISON")
print("=" * 70)

comparison = {
    "BiLSTM_Attention": {
        "model": bilstm_model,
        "roc_auc": bilstm_roc_auc,
        "pr_auc": bilstm_pr_auc,
        "train_time": bilstm_train_time,
        "params": bilstm_model.count_params(),
        "predictions_test": bilstm_pred_test,
    },
    "BiGRU_Attention": {
        "model": bigru_model,
        "roc_auc": bigru_roc_auc,
        "pr_auc": bigru_pr_auc,
        "train_time": bigru_train_time,
        "params": bigru_model.count_params(),
        "predictions_test": bigru_pred_test,
    },
}

print(f"\n   {'Model':<20} {'ROC-AUC':>10} {'PR-AUC':>10} {'Time':>10} {'Params':>10}")
print(f"   {'─' * 60}")
for name, info in comparison.items():
    print(
        f"   {name:<20} {info['roc_auc']:>10.4f} {info['pr_auc']:>10.4f} "
        f"{info['train_time']:>8.1f}s {info['params']:>10,}"
    )

# Select winner by PR-AUC
winner_name = max(comparison, key=lambda k: comparison[k]["pr_auc"])
winner_info = comparison[winner_name]
winner_model = winner_info["model"]

print(f"\n   🏆 WINNER: {winner_name} (PR-AUC: {winner_info['pr_auc']:.4f})")

# Save both models (for documentation) and mark the winner
bilstm_model.save(f"{V4_MODEL_DIR}/paysim_v4_bilstm.keras")
bigru_model.save(f"{V4_MODEL_DIR}/paysim_v4_bigru.keras")
winner_model.save(f"{V4_MODEL_DIR}/paysim_v4_sequential_winner.keras")

# Save comparison results
comparison_results = {
    "winner": winner_name,
    "models": {
        name: {
            "roc_auc": info["roc_auc"],
            "pr_auc": info["pr_auc"],
            "train_time_seconds": info["train_time"],
            "parameters": info["params"],
        }
        for name, info in comparison.items()
    },
}
with open(f"{V4_MODEL_DIR}/sequential_comparison.json", "w") as f:
    json.dump(comparison_results, f, indent=2)

print(f"   Both models saved. Winner marked as 'sequential_winner'.")


# ══════════════════════════════════════════════════════
# PHASE 8: COMPUTING SEQUENTIAL SCORES (20th FEATURE)
# ══════════════════════════════════════════════════════
"""
The winning sequential model's fraud probability becomes the 20th feature
for XGBoost and Random Forest.

This is the same idea as V3's 19th feature (AE reconstruction error):
  - The AE captures ANOMALY patterns → feeds XGBoost
  - The LSTM/GRU captures SEQUENTIAL patterns → feeds XGBoost
  - XGBoost learns to combine ALL signal sources for best predictions

For transactions at the start of the dataset (< SEQ_LENGTH), we can't
form a full sequence. For these, we use a DEFAULT SCORE of 0.0
(= low fraud probability), since we have no sequential context.
"""
print("\n📊 Phase 8: Computing sequential scores (20th feature)...")


def compute_sequential_scores(model, X_scaled, seq_length, batch_size=512):
    """
    Compute per-transaction sequential fraud scores.

    For transaction at index i:
      - If i >= seq_length: use window [i-seq_length : i]
      - If i < seq_length: pad with zeros (no history available)

    Returns: (n_transactions,) array of fraud probabilities
    """
    n_samples = len(X_scaled)
    n_features = X_scaled.shape[1]
    scores = np.zeros(n_samples, dtype=np.float32)

    # Create all sequences at once (vectorized)
    sequences = []
    valid_indices = []

    for i in range(seq_length, n_samples):
        sequences.append(X_scaled[i - seq_length : i])
        valid_indices.append(i)

    if len(sequences) > 0:
        sequences = np.array(sequences, dtype=np.float32)
        preds = model.predict(sequences, batch_size=batch_size, verbose=0).ravel()
        scores[valid_indices] = preds

    return scores


# Compute scores for each split
seq_scores_train = compute_sequential_scores(winner_model, X_train_scaled, SEQ_LENGTH)
seq_scores_val = compute_sequential_scores(winner_model, X_val_scaled, SEQ_LENGTH)
seq_scores_test = compute_sequential_scores(winner_model, X_test_scaled, SEQ_LENGTH)

print(f"   Sequential score stats (train):")
print(f"     Legitimate mean: {seq_scores_train[y_train == 0].mean():.6f}")
print(f"     Fraud mean:      {seq_scores_train[y_train == 1].mean():.6f}")


# ══════════════════════════════════════════════════════
# PHASE 9: BUILD 20-FEATURE DATASET & TRAIN XGBoost + RF
# ══════════════════════════════════════════════════════
"""
Now we stack the three signal sources into a 20-feature dataset:

  Feature 1-18:  Original engineered features (amount, balance, velocity, etc.)
  Feature 19:    AE reconstruction error (anomaly signal)
  Feature 20:    Sequential model score (temporal pattern signal)

XGBoost and Random Forest learn to COMBINE these signals optimally.
The AE catches anomalous individual transactions.
The LSTM/GRU catches suspicious sequential patterns.
XGBoost/RF combine everything into a final prediction.
"""
print("\n🌲 Phase 9: Training XGBoost + RF on 20 features...")

# Stack: 18 scaled features + ae_error + seq_score → 20 features
X_train_20 = np.column_stack([X_train_scaled, ae_err_train, seq_scores_train])
X_val_20 = np.column_stack([X_val_scaled, ae_err_val, seq_scores_val])
X_test_20 = np.column_stack([X_test_scaled, ae_err_test, seq_scores_test])

# Create and fit a new scaler for the full 20-feature set
scaler_20 = StandardScaler()
X_train_20_s = scaler_20.fit_transform(X_train_20)
X_val_20_s = scaler_20.transform(X_val_20)
X_test_20_s = scaler_20.transform(X_test_20)

# Save 20-feature scaler
joblib.dump(scaler_20, f"{V4_MODEL_DIR}/paysim_v4_scaler.pkl")

# Extended feature list
features_20 = FEATURES + ["ae_recon_error", "sequential_score"]
joblib.dump(features_20, f"{V4_MODEL_DIR}/paysim_v4_features_20.pkl")

print(f"   20-feature dataset shape: {X_train_20_s.shape}")

# ── Train Random Forest ──
print("\n   Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=RF_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train_20_s, y_train)
joblib.dump(rf, f"{V4_MODEL_DIR}/paysim_v4_rf.pkl")

rf_prob_test = rf.predict_proba(X_test_20_s)[:, 1]
rf_auc = roc_auc_score(y_test, rf_prob_test)
rf_prauc = average_precision_score(y_test, rf_prob_test)

print(f"   RF ROC-AUC: {rf_auc:.4f} | PR-AUC: {rf_prauc:.4f}")

# ── Train XGBoost ──
print("\n   Training XGBoost...")
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=XGB_ESTIMATORS,
    max_depth=XGB_MAX_DEPTH,
    learning_rate=XGB_LEARNING_RATE,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
)
xgb_model.fit(X_train_20_s, y_train)
joblib.dump(xgb_model, f"{V4_MODEL_DIR}/paysim_v4_xgb.pkl")

xgb_prob_test = xgb_model.predict_proba(X_test_20_s)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_prob_test)
xgb_prauc = average_precision_score(y_test, xgb_prob_test)

print(f"   XGB ROC-AUC: {xgb_auc:.4f} | PR-AUC: {xgb_prauc:.4f}")


# ══════════════════════════════════════════════════════
# PHASE 10: THRESHOLD OPTIMIZATION
# ══════════════════════════════════════════════════════
"""
Path A Decision Threshold
─────────────────────────
The ensemble prediction: P = 0.5 * P_XGB + 0.5 * P_RF

We need a threshold T such that:
  - P >= T → AUTO-BLOCK (confirmed fraud)
  - P < T  → pass to Path B for anomaly check

We optimize T on the VALIDATION set to find the highest precision
while maintaining at least TARGET_RECALL (80%).

WHY not optimize on test set:
  Test set is held out for FINAL evaluation only. If we tuned on
  test data, our reported metrics would be overoptimistic.
"""
print("\n🎯 Phase 10: Threshold optimization (Path A)...")

# Ensemble predictions on validation set
xgb_prob_val = xgb_model.predict_proba(X_val_20_s)[:, 1]
rf_prob_val = rf.predict_proba(X_val_20_s)[:, 1]
ensemble_prob_val = 0.5 * xgb_prob_val + 0.5 * rf_prob_val

# Find threshold: highest precision with recall >= TARGET_RECALL
precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_val, ensemble_prob_val)
precision_arr, recall_arr = precision_arr[:-1], recall_arr[:-1]

valid_idx = np.where(recall_arr >= TARGET_RECALL)[0]
if len(valid_idx) > 0:
    best_idx = valid_idx[-1]  # Highest threshold that still meets recall target
    block_threshold = float(thresholds_arr[best_idx])
else:
    # Fallback: best F1 score
    f1_arr = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-12)
    best_idx = np.argmax(f1_arr)
    block_threshold = float(thresholds_arr[best_idx])
    print("   ⚠️  Could not meet recall target, using best F1 threshold")

np.save(f"{V4_MODEL_DIR}/paysim_v4_threshold.npy", np.array([block_threshold]))
print(f"   Block threshold: {block_threshold:.6f}")

# Save ensemble weights
weights = np.array([0.5, 0.5])
np.save(f"{V4_MODEL_DIR}/paysim_v4_weights.npy", weights)

# Compute sequential anomaly threshold for Path B
# Using the winner model's scores on validation legitimate transactions
seq_scores_val_full = compute_sequential_scores(winner_model, X_val_scaled, SEQ_LENGTH)
seq_threshold = np.percentile(seq_scores_val_full[y_val == 0], 95)
np.save(f"{V4_MODEL_DIR}/paysim_v4_seq_threshold.npy", np.array([seq_threshold]))
print(f"   Sequential anomaly threshold (95th pct): {seq_threshold:.6f}")


# ══════════════════════════════════════════════════════
# PHASE 11: ISOLATION FOREST (SAME AS V3)
# ══════════════════════════════════════════════════════
"""
Isolation Forest — detects outliers by random partitioning.

HOW it works:
  - Randomly selects a feature and a split value
  - Outliers require FEWER splits to isolate (they're "far" from normal data)
  - Anomaly score = average path length across all trees

In Path B, it provides a THIRD anomaly signal alongside AE and LSTM.
"""
print("\n🌳 Phase 11: Training Isolation Forest...")

iforest = IsolationForest(
    n_estimators=IFOREST_ESTIMATORS,
    contamination=IFOREST_CONTAMINATION,
    random_state=42,
    n_jobs=-1,
)
iforest.fit(X_train_scaled[y_train == 0])  # Train on legitimate only
joblib.dump(iforest, f"{V4_MODEL_DIR}/paysim_v4_iforest.pkl")

iforest_pred_test = iforest.predict(X_test_scaled)
iforest_anomalies = (iforest_pred_test == -1).sum()
print(f"   IForest flagged {iforest_anomalies:,} anomalies in test set ({iforest_anomalies / len(y_test):.2%})")

# Save sequence length config
joblib.dump(SEQ_LENGTH, f"{V4_MODEL_DIR}/paysim_v4_seq_length.pkl")


# ══════════════════════════════════════════════════════
# PHASE 12: FINAL EVALUATION
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(" V4 FINAL EVALUATION (TEST SET)")
print("=" * 70)

# Path A: Auto-block
ensemble_prob_test = 0.5 * xgb_prob_test + 0.5 * rf_prob_test
path_a_block = ensemble_prob_test >= block_threshold
y_pred_path_a = path_a_block.astype(int)

# Path B: Flag for review
ae_flag = ae_err_test >= ae_threshold
iforest_flag = iforest.predict(X_test_scaled) == -1
seq_flag = seq_scores_test >= seq_threshold
path_b_review = (ae_flag | iforest_flag | seq_flag) & ~path_a_block

# Combined decisions
y_pred_combined = (path_a_block | path_b_review).astype(int)

# Metrics
cm = confusion_matrix(y_test, y_pred_path_a)
tn, fp, fn, tp = cm.ravel()

print(f"\n   PATH A (Auto-Block) — Threshold: {block_threshold:.4f}")
print(f"   ─────────────────────────────────────")
print(f"   Recall:     {tp / (tp + fn):.4f} ({tp}/{tp + fn})")
print(f"   Precision:  {tp / (tp + fp):.4f} ({tp}/{tp + fp})")
print(f"   F1-Score:   {2 * tp / (2 * tp + fp + fn):.4f}")
print(f"   ROC-AUC:    {roc_auc_score(y_test, ensemble_prob_test):.4f}")
print(f"   Alert rate: {(tp + fp) / len(y_test):.2%}")
print(f"\n   Confusion Matrix:")
print(f"     TN={tn:,}  FP={fp:,}")
print(f"     FN={fn:,}  TP={tp:,}")

print(f"\n   PATH B (Flag for Review)")
print(f"   ─────────────────────────────────────")
print(f"   AE flags:       {ae_flag.sum():,}")
print(f"   IForest flags:  {(iforest_flag).sum():,}")
print(f"   Sequential flags: {seq_flag.sum():,}")
print(f"   Total reviews (excl. blocked): {path_b_review.sum():,}")

# How many frauds Path B catches that Path A missed
path_a_missed_frauds = (y_test == 1) & ~path_a_block
path_b_caught = path_b_review & path_a_missed_frauds
if path_a_missed_frauds.sum() > 0:
    novel_catch_rate = path_b_caught.sum() / path_a_missed_frauds.sum()
    print(f"   Novel fraud catch rate: {novel_catch_rate:.1%} ({path_b_caught.sum()}/{path_a_missed_frauds.sum()} missed frauds)")

print(f"\n   Classification Report (Path A):")
print(classification_report(y_test, y_pred_path_a, digits=4))

# Save final results
final_results = {
    "v4_path_a": {
        "threshold": block_threshold,
        "recall": float(tp / (tp + fn)),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)),
        "roc_auc": float(roc_auc_score(y_test, ensemble_prob_test)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    },
    "v4_path_b": {
        "ae_threshold": float(ae_threshold),
        "seq_threshold": float(seq_threshold),
        "ae_flags": int(ae_flag.sum()),
        "iforest_flags": int((iforest_flag).sum()),
        "seq_flags": int(seq_flag.sum()),
        "total_reviews": int(path_b_review.sum()),
        "novel_catch_rate": float(novel_catch_rate) if path_a_missed_frauds.sum() > 0 else 0.0,
    },
    "sequential_comparison": comparison_results,
    "features_count": 20,
    "data_fraction_used": DATA_PCT,
}

with open(f"{V4_MODEL_DIR}/v4_training_results.json", "w") as f:
    json.dump(final_results, f, indent=2)


# ══════════════════════════════════════════════════════
# CLEANUP
# ══════════════════════════════════════════════════════
del df
gc.collect()

print("\n" + "=" * 70)
print(" ✨ V4 HYBRID PIPELINE COMPLETE!")
print("=" * 70)
print(f"\n   All artifacts saved to: {V4_MODEL_DIR}/")
print(f"   Sequential winner: {winner_name}")
print(f"   Features: 20 (18 base + ae_error + seq_score)")
print(f"\n   V3 models: UNTOUCHED ✅")
print(f"\n   Next steps:")
print(f"     1. Run evaluation: python experiments/evaluate_v4_vs_v3.py")
print(f"     2. If V4 is better: update model_loader.py to use V4")
print(f"     3. If V4 is worse: delete {V4_MODEL_DIR}/ — no damage done")
print("=" * 70)
