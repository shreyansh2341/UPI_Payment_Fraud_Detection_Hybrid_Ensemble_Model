import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

# ======================================================
# CONFIG
# ======================================================
MODEL_PATH = "models/fraud_lstm_model_focal.keras"
TARGET_RECALL = 0.80

# Cost assumptions (illustrative)
FN_COST = 10_000   # Missed fraud cost
FP_COST = 100      # False alert cost

# ======================================================
# LOAD MODEL
# ======================================================
print("Loading trained LSTM model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# ======================================================
# LOAD DATA
# ======================================================
X = np.load("data/lstm_sequences/X.npy")
y = np.load("data/lstm_sequences/y.npy")

n = len(X)
test_start = int(0.85 * n)

X_test = X[test_start:]
y_test = y[test_start:]

print(f"Test samples: {len(y_test)}")

# ======================================================
# PREDICT PROBABILITIES
# ======================================================
y_prob = model.predict(X_test, batch_size=512).ravel()

# ======================================================
# PRECISION-RECALL CURVE
# ======================================================
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Align lengths (IMPORTANT)
precision = precision[:-1]
recall = recall[:-1]

# ======================================================
# THRESHOLD SELECTION (PR-AWARE, SAFE)
# ======================================================
valid_idx = np.where(recall >= TARGET_RECALL)[0]

if len(valid_idx) == 0:
    raise RuntimeError("No threshold achieves target recall")

best_idx = valid_idx[-1]  # highest threshold meeting recall
best_threshold = thresholds[best_idx]

# ======================================================
# APPLY THRESHOLD
# ======================================================
y_pred = (y_prob >= best_threshold).astype(int)

# ======================================================
# CONFUSION MATRIX
# ======================================================
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# ======================================================
# SAFE METRICS
# ======================================================
recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0

threshold_percentile = (y_prob < best_threshold).mean() * 100

# ======================================================
# PRINT RESULTS
# ======================================================
print("\nSelected Threshold Analysis")
print("-" * 40)
print(f"Threshold            : {best_threshold:.6f}")
print(f"Recall               : {recall_val:.4f}")
print(f"Precision            : {precision_val:.4f}")
print(f"Threshold percentile : {threshold_percentile:.2f}%")

print("\nConfusion Matrix")
print("-" * 40)
print(cm)

print("\nClassification Report")
print("-" * 40)
print(classification_report(y_test, y_pred, digits=4))

# ======================================================
# BUSINESS METRICS
# ======================================================
alert_rate = (tp + fp) / len(y_test)
total_cost = FN_COST * fn + FP_COST * fp

print("\nBusiness-Oriented Metrics")
print("-" * 40)
print(f"Alert rate            : {alert_rate:.4%}")
print(f"Alerts per 10,000 txns: {alert_rate * 10_000:.2f}")
print(f"Frauds caught         : {tp} / {tp + fn}")
print(f"Fraud capture rate    : {recall_val:.2%}")
print(f"Estimated loss (₹)    : {total_cost:,}")
