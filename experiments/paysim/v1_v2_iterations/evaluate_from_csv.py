import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc
)

# =========================
# CONFIG
# =========================
CSV_PATH = "Fraud_Detection_Model_Paysim_CC/data/paysim_predictions.csv"
TARGET_COL = "isFraud"
PRED_COL = "fraud_prediction"
SCORE_COL = "fraud_score"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)

y_true = df[TARGET_COL].values
y_pred = df[PRED_COL].values
y_score = df[SCORE_COL].values

print("📊 Dataset size:", len(df))
print("Fraud rate:", y_true.mean())
print()

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)
print("CONFUSION MATRIX")
print(cm)
print()

# =========================
# CLASSIFICATION REPORT
# =========================
print("CLASSIFICATION REPORT")
print(classification_report(y_true, y_pred, digits=4))

# =========================
# ROC-AUC
# =========================
roc_auc = roc_auc_score(y_true, y_score)
print(f"ROC-AUC: {roc_auc:.4f}")

# =========================
# PR-AUC (MOST IMPORTANT)
# =========================
precision, recall, _ = precision_recall_curve(y_true, y_score)
pr_auc = auc(recall, precision)
print(f"PR-AUC: {pr_auc:.4f}")
