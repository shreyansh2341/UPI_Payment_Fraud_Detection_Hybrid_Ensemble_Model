import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

DATA = Path("Fraud_Detection_Model_Paysim_CC/data/paysim_augmented_with_frauds.csv")
MODELS = Path("Fraud_Detection_Model_Paysim_CC/models/stage2_experiment")

# Load artifacts
xgb = joblib.load(MODELS / "paysim_xgb_stage2.pkl")
rf = joblib.load(MODELS / "paysim_rf_stage2.pkl")
scaler = joblib.load(MODELS / "paysim_stage2_scaler.pkl")
features = joblib.load(MODELS / "paysim_stage2_features.pkl")
threshold = float(np.load(MODELS / "paysim_stage2_threshold.npy"))

df = pd.read_csv(DATA)
X = df[features]
y = df["isFraud"]

X_scaled = scaler.transform(X)

def evaluate(model, name):
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= threshold).astype(int)

    print(f"\n===== {name} =====")
    print("ROC-AUC:", roc_auc_score(y, probs))
    print("Confusion Matrix:\n", confusion_matrix(y, preds))
    print(classification_report(y, preds, target_names=["Legit", "Fraud"]))

evaluate(xgb, "XGBoost")
evaluate(rf, "Random Forest")

# Ensemble
probs_ens = 0.6 * xgb.predict_proba(X_scaled)[:,1] + \
            0.4 * rf.predict_proba(X_scaled)[:,1]
preds_ens = (probs_ens >= threshold).astype(int)

print("\n===== Ensemble =====")
print("ROC-AUC:", roc_auc_score(y, probs_ens))
print("Confusion Matrix:\n", confusion_matrix(y, preds_ens))
print(classification_report(y, preds_ens, target_names=["Legit", "Fraud"]))
