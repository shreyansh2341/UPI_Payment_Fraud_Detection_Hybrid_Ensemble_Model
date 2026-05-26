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

import xgboost as xgb

# ======================================================
# CONFIG
# ======================================================
DATA_PATH = "data/cleaned_creditcard.csv"
MODEL_DIR = "models"
TARGET_RECALL = 0.80

os.makedirs(MODEL_DIR, exist_ok=True)

# # ======================================================
# # LOAD DATA
# # ======================================================
# print("Loading credit card dataset...")
# df = pd.read_csv(DATA_PATH)
# df.columns = df.columns.str.lower()

# TARGET = "class"

# X = df.drop(columns=[TARGET])
# y = df[TARGET].values

# print("Total samples :", len(df))
# print("Fraud ratio   :", y.mean())

# ======================================================
# LOAD DATA
# ======================================================
print("Loading credit card dataset...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower()

TARGET = "isfraud"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found")

X = df.drop(columns=[TARGET])
y = df[TARGET].values

print("Total samples :", len(df))
print("Fraud ratio   :", y.mean())
print("Number of features:", X.shape[1])


# ======================================================
# TRAIN / VAL / TEST SPLIT (TIME-AWARE IF POSSIBLE)
# ======================================================
n = len(df)
train_end = int(0.7 * n)
val_end   = int(0.85 * n)

X_train, y_train = X.iloc[:train_end], y[:train_end]
X_val,   y_val   = X.iloc[train_end:val_end], y[train_end:val_end]
X_test,  y_test  = X.iloc[val_end:], y[val_end:]

# ======================================================
# SCALE FEATURES (FIT ON TRAIN ONLY)
# ======================================================
print("Scaling features...")
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

joblib.dump(scaler, f"{MODEL_DIR}/cc_scaler.pkl")
joblib.dump(list(X.columns), f"{MODEL_DIR}/cc_features.pkl")

# ======================================================
# XGBOOST
# ======================================================
print("Training XGBoost (Credit Card)...")
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train)
joblib.dump(xgb_model, f"{MODEL_DIR}/cc_xgb_model.pkl")

# ======================================================
# EVALUATION (RANKING)
# ======================================================
y_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]

print("\n=== CREDIT CARD RANKING PERFORMANCE ===")
print("ROC-AUC :", roc_auc_score(y_test, y_prob))
print("PR-AUC  :", average_precision_score(y_test, y_prob))

# ======================================================
# THRESHOLD TUNING (RECALL-CONSTRAINED)
# ======================================================
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
precision, recall = precision[:-1], recall[:-1]

valid_idx = np.where(recall >= TARGET_RECALL)[0]
best_idx = valid_idx[-1]
best_threshold = thresholds[best_idx]

y_pred = (y_prob >= best_threshold).astype(int)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n=== CREDIT CARD DECISION METRICS ===")
print(f"Threshold  : {best_threshold:.6f}")
print(f"Recall     : {tp / (tp + fn):.4f}")
print(f"Precision  : {tp / (tp + fp):.4f}")
print(f"Alert rate : {(tp + fp) / len(y_test):.2%}")

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(classification_report(y_test, y_pred, digits=4))

np.save(f"{MODEL_DIR}/cc_threshold.npy",
        np.array([best_threshold]))

# ======================================================
# CLEANUP
# ======================================================
del df
gc.collect()

print("\nCredit Card Stage-2 model complete.")
