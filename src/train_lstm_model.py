import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score

# ======================================================
# FOCAL LOSS (FRAUD-TUNED + NUMERICALLY STABLE)
# ======================================================
def focal_loss(alpha=0.75, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        # numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)

        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1 - p_t, gamma)

        return alpha_factor * modulating_factor * bce

    return loss

# ======================================================
# LOAD DATA
# ======================================================
print("Loading LSTM sequences...")
X = np.load("data/lstm_sequences/X.npy")
y = np.load("data/lstm_sequences/y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Fraud ratio:", y.mean())

# ======================================================
# TIME-BASED SPLIT (NO LEAKAGE)
# ======================================================
n = len(X)
train_end = int(0.70 * n)
val_end   = int(0.85 * n)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val     = X[train_end:val_end], y[train_end:val_end]
X_test, y_test   = X[val_end:], y[val_end:]

print("Train size:", X_train.shape)
print("Val size  :", X_val.shape)
print("Test size :", X_test.shape)

# ======================================================
# MODEL ARCHITECTURE (CLEAN + PRODUCTION SAFE)
# ======================================================
model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    LSTM(64),
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

# ======================================================
# CALLBACKS (PR-AUC DRIVEN)
# ======================================================
early_stop = EarlyStopping(
    monitor="val_pr_auc",
    mode="max",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_pr_auc",
    mode="max",
    factor=0.5,
    patience=3,
    min_lr=1e-5
)

# ======================================================
# TRAIN (NO CLASS WEIGHTS WITH FOCAL LOSS)
# ======================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=256,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ======================================================
# TEST EVALUATION
# ======================================================
print("\nEvaluating on test set...")
y_prob = model.predict(X_test, batch_size=512).ravel()

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc  = average_precision_score(y_test, y_prob)

print("Test ROC-AUC :", roc_auc)
print("Test PR-AUC  :", pr_auc)

# ======================================================
# SAVE MODEL (MODERN FORMAT)
# ======================================================
model.save("models/fraud_lstm_model_focal.keras")
print("\nModel saved to models/fraud_lstm_model_focal.keras")
