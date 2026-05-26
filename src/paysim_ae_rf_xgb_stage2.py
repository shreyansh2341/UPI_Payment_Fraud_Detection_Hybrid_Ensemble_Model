import numpy as np
import pandas as pd
import joblib
import os
import gc

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ======================================================
# CONFIG
# ======================================================
DATA_PATH = "data/cleaned_paysim_lstm.csv"
MODEL_DIR = "models"

AE_EPOCHS = 30
AE_BATCH  = 1024
TARGET_RECALL = 0.80   # business constraint

os.makedirs(MODEL_DIR, exist_ok=True)

# ======================================================
# LOAD DATA
# ======================================================
print("Loading PaySim dataset...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower()

df = df.sort_values("step").reset_index(drop=True)

TARGET = "isfraud"
DROP_COLS = ["isfraud", "nameorig", "namedest", "step", "datetime"]

FEATURES = [c for c in df.columns if c not in DROP_COLS]

X = df[FEATURES].values
y = df[TARGET].values

# ======================================================
# TIME-AWARE SPLIT
# ======================================================
n = len(df)
train_end = int(0.7 * n)
val_end   = int(0.85 * n)

X_train, y_train = X[:train_end], y[:train_end]
X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
X_test,  y_test  = X[val_end:], y[val_end:]

print("Train / Val / Test sizes:",
      X_train.shape, X_val.shape, X_test.shape)

# ======================================================
# SCALE FEATURES (FIT ON TRAIN ONLY)
# ======================================================
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

joblib.dump(scaler, f"{MODEL_DIR}/paysim_stage2_scaler.pkl")
joblib.dump(FEATURES, f"{MODEL_DIR}/paysim_stage2_features.pkl")

# ======================================================
# AUTOENCODER (TRAIN ON NORMAL ONLY)
# ======================================================
print("Training Autoencoder on normal transactions...")
X_train_normal = X_train_scaled[y_train == 0]

input_dim = X_train_normal.shape[1]

inp = Input(shape=(input_dim,))
x = Dense(64, activation="relu")(inp)
x = Dense(32, activation="relu")(x)
encoded = Dense(16, activation="relu")(x)

x = Dense(32, activation="relu")(encoded)
x = Dense(64, activation="relu")(x)
out = Dense(input_dim, activation="linear")(x)

ae = Model(inp, out)
ae.compile(optimizer="adam", loss="mse")

ae.fit(
    X_train_normal,
    X_train_normal,
    epochs=AE_EPOCHS,
    batch_size=AE_BATCH,
    validation_split=0.1,
    shuffle=True,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)

ae.save(f"{MODEL_DIR}/paysim_ae_model.keras")

# ======================================================
# AE RECONSTRUCTION ERROR (LOG-SCALED)
# ======================================================
def ae_error(model, X):
    recon = model.predict(X, batch_size=2048)
    err = np.mean(np.square(X - recon), axis=1)
    return np.log1p(err)

ae_train = ae_error(ae, X_train_scaled)
ae_val   = ae_error(ae, X_val_scaled)
ae_test  = ae_error(ae, X_test_scaled)

# ======================================================
# SUPERVISED DATA (ADD AE FEATURE)
# ======================================================
X_train_sup = np.column_stack([X_train_scaled, ae_train])
X_val_sup   = np.column_stack([X_val_scaled, ae_val])
X_test_sup  = np.column_stack([X_test_scaled, ae_test])

# ======================================================
# RANDOM FOREST
# ======================================================
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train_sup, y_train)
joblib.dump(rf, f"{MODEL_DIR}/paysim_rf_stage2.pkl")

rf_prob = rf.predict_proba(X_test_sup)[:, 1]

# ======================================================
# XGBOOST
# ======================================================
print("Training XGBoost...")
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42
)

xgb_model.fit(X_train_sup, y_train)
joblib.dump(xgb_model, f"{MODEL_DIR}/paysim_xgb_stage2.pkl")

xgb_prob = xgb_model.predict_proba(X_test_sup)[:, 1]

# ======================================================
# METRICS (RANKING)
# ======================================================
print("\n=== STAGE-2 RANKING PERFORMANCE ===")
print("RF  ROC-AUC :", roc_auc_score(y_test, rf_prob))
print("RF  PR-AUC  :", average_precision_score(y_test, rf_prob))
print("XGB ROC-AUC :", roc_auc_score(y_test, xgb_prob))
print("XGB PR-AUC  :", average_precision_score(y_test, xgb_prob))

# ======================================================
# THRESHOLD TUNING (XGBOOST – FINAL DECISION MODEL)
# ======================================================
precision, recall, thresholds = precision_recall_curve(y_test, xgb_prob)
precision, recall = precision[:-1], recall[:-1]

valid_idx = np.where(recall >= TARGET_RECALL)[0]
best_idx = valid_idx[-1]
best_threshold = thresholds[best_idx]

y_pred = (xgb_prob >= best_threshold).astype(int)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n=== STAGE-2 DECISION METRICS (XGBoost) ===")
print(f"Threshold  : {best_threshold:.6f}")
print(f"Recall     : {tp / (tp + fn):.4f}")
print(f"Precision  : {tp / (tp + fp):.4f}")
print(f"Alert rate : {(tp + fp) / len(y_test):.2%}")

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(classification_report(y_test, y_pred, digits=4))

np.save(f"{MODEL_DIR}/paysim_stage2_threshold.npy",
        np.array([best_threshold]))

# ======================================================
# CLEANUP
# ======================================================
del df
gc.collect()

print("\nStage-2 PaySim pipeline complete.")
