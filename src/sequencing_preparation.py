import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import gc
import joblib

# ======================================================
# CONFIG (OPTIMIZED FOR YOUR DEVICE)
# ======================================================
DATA_PATH = "data/cleaned_paysim_lstm.csv"
SAVE_DIR = "data/lstm_sequences"
MODEL_DIR = "models"

SEQUENCE_LENGTH = 5        # Short, effective temporal context
MAX_SEQUENCES = 300_000    # RAM-safe

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ======================================================
# LOAD DATA
# ======================================================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

# ======================================================
# DROP UNUSED COLUMNS
# ======================================================
if 'datetime' in df.columns:
    df.drop(columns=['datetime'], inplace=True)

# ======================================================
# TIME FEATURES FROM PaySim STEP
# ======================================================
df['hour'] = df['step'] % 24
df['dayofweek'] = (df['step'] // 24) % 7
df['is_weekend'] = (df['dayofweek'] >= 5).astype(np.int8)

# ======================================================
# SORT BY GLOBAL TIME (CRITICAL)
# ======================================================
df = df.sort_values('step').reset_index(drop=True)

# ======================================================
# GLOBAL ROLLING / BEHAVIORAL FEATURES
# (SYSTEM-WIDE TEMPORAL MEMORY)
# ======================================================
print("Creating global behavioral memory features...")

df['rolling_mean_amount'] = (
    df['amount']
    .rolling(window=SEQUENCE_LENGTH, min_periods=1)
    .mean()
)

df['rolling_std_amount'] = (
    df['amount']
    .rolling(window=SEQUENCE_LENGTH, min_periods=1)
    .std()
    .fillna(0)
)

df['cumulative_amount'] = np.log1p(df['amount'].cumsum())

df['balance_mismatch_rate'] = (
    df['has_balance_mismatch']
    .rolling(window=SEQUENCE_LENGTH, min_periods=1)
    .mean()
)

# ======================================================
# OPTIONAL STABILITY: CLIP HEAVY-TAILED FEATURES
# ======================================================
for col in ['cumulative_amount', 'rolling_std_amount']:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower, upper)

# ======================================================
# EXPLICIT BOOLEAN CASTING
# ======================================================
df['upi_type_upi_payment'] = df['upi_type_upi_payment'].astype(np.int8)
df['upi_type_upi_transfer'] = df['upi_type_upi_transfer'].astype(np.int8)
df['has_balance_mismatch'] = df['has_balance_mismatch'].astype(np.int8)

# ======================================================
# FEATURES FOR LSTM
# ======================================================
FEATURES = [
    'amount',
    'hour',
    'dayofweek',
    'is_weekend',

    'rolling_mean_amount',
    'rolling_std_amount',
    'cumulative_amount',
    'balance_mismatch_rate',

    'errorbalanceorig',
    'errorbalancedest',
    'upi_type_upi_payment',
    'upi_type_upi_transfer'
]

TARGET = 'isfraud'

# ======================================================
# DEFENSIVE FEATURE CHECK
# ======================================================
missing = set(FEATURES) - set(df.columns)
if missing:
    raise ValueError(f"Missing required features: {missing}")

# ======================================================
# SCALE FEATURES (SAVE FOR INFERENCE)
# ======================================================
print("Scaling features...")
scaler = StandardScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])

joblib.dump(scaler, os.path.join(MODEL_DIR, "lstm_scaler.pkl"))
joblib.dump(FEATURES, os.path.join(MODEL_DIR, "lstm_features.pkl"))
joblib.dump(SEQUENCE_LENGTH, os.path.join(MODEL_DIR, "lstm_seq_len.pkl"))

# ======================================================
# CREATE GLOBAL TEMPORAL LSTM SEQUENCES
# ======================================================
print("Creating global LSTM sequences...")

values = df[FEATURES].values
labels = df[TARGET].values

X, y = [], []

for i in range(len(df) - SEQUENCE_LENGTH):
    X.append(values[i:i + SEQUENCE_LENGTH])
    y.append(labels[i + SEQUENCE_LENGTH])  # predict next transaction

# ======================================================
# CONVERT TO NUMPY
# ======================================================
X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int8)

# ======================================================
# LIMIT TOTAL SEQUENCES (RAM SAFE)
# ======================================================
print("Applying sequence cap...")
if len(X) > MAX_SEQUENCES:
    idx = np.random.choice(len(X), MAX_SEQUENCES, replace=False)
    X = X[idx]
    y = y[idx]

# ======================================================
# SHUFFLE SEQUENCES
# ======================================================
print("Shuffling sequences...")
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

# ======================================================
# SAVE OUTPUT
# ======================================================
print("Saving sequences...")
np.save(os.path.join(SAVE_DIR, "X.npy"), X)
np.save(os.path.join(SAVE_DIR, "y.npy"), y)

# ======================================================
# CLEANUP
# ======================================================
del df
gc.collect()

# ======================================================
# STATS
# ======================================================
print("Done.")
print(f"X shape       : {X.shape}")
print(f"y shape       : {y.shape}")
print(f"Fraud samples : {y.sum()}")
print(f"Fraud ratio   : {y.mean():.6f}")
