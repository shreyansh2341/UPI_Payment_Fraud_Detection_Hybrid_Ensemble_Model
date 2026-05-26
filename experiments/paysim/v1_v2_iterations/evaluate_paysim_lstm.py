import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc

# ===============================
# PATHS
# ===============================
X_PATH = "data/lstm_sequences/X.npy"
Y_PATH = "data/lstm_sequences/y.npy"

MODEL_PATH = "models/fraud_lstm_model_focal.keras"

THRESHOLD = 0.5   # baseline, we can tune later

print("Evaluating PaySim LSTM (Sequential Model)")

# ===============================
# LOAD DATA
# ===============================
X = np.load(X_PATH)
y = np.load(Y_PATH)

print("Sequence shape:", X.shape)
print("Labels shape  :", y.shape)

# ===============================
# LOAD MODEL
# ===============================
model = tf.keras.models.load_model(MODEL_PATH,
                                   compile=False )

# ===============================
# PREDICTION
# ===============================
probs = model.predict(X, batch_size=1024, verbose=1).ravel()
y_pred = (probs >= THRESHOLD).astype(int)

# ===============================
# METRICS
# ===============================
print("\nLSTM CONFUSION MATRIX")
print(confusion_matrix(y, y_pred))

print("\nLSTM CLASSIFICATION REPORT")
print(classification_report(y, y_pred))

# ===============================
# PR-AUC (important for imbalance)
# ===============================
precision, recall, _ = precision_recall_curve(y, probs)
pr_auc = auc(recall, precision)

print(f"\nLSTM PR-AUC: {pr_auc:.4f}")
