import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, classification_report,
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)

# ---------- CONFIG ----------
DATASET = "paysim"  # or "creditcard"
OUTPUT_DIR = Path("artifacts")
DATA_DIR = Path("data")
PAYSIM_FILE = DATA_DIR / "cleaned_paysim.csv"
CREDIT_FILE = DATA_DIR / "cleaned_creditcard.csv"
SAVE_PLOTS = True  # Set True to save plots as PNG
# ----------------------------

# ---------- LOAD MODELS ----------
print("Loading models and preprocessors...")
ae = joblib.load(OUTPUT_DIR / f"{DATASET}_ae_model.keras")
preproc = joblib.load(OUTPUT_DIR / f"{DATASET}_ae_preproc.joblib")
sup_preproc = joblib.load(OUTPUT_DIR / f"{DATASET}_sup_preproc.joblib")
rf = joblib.load(OUTPUT_DIR / f"{DATASET}_rf_ae.joblib")
xgb = joblib.load(OUTPUT_DIR / f"{DATASET}_xgb_ae.joblib")
thresh = np.load(OUTPUT_DIR / f"{DATASET}_ae_thresh.npy")[0]

# ---------- LOAD DATA ----------
df = pd.read_csv(PAYSIM_FILE if DATASET == "paysim" else CREDIT_FILE)
df = df.sample(10000, random_state=42).reset_index(drop=True)
X = df.drop(columns=['isFraud'])
y = df['isFraud']

# ---------- COMPUTE AE FEATURES AGAIN ----------
print("Computing AE features (reconstruction error & anomaly flag)...")
Xp = preproc.transform(X)
recon = ae.predict(Xp, verbose=0)
errs = np.mean((Xp - recon)**2, axis=1)
flags = (errs > thresh).astype(int)

X_aug = X.copy()
X_aug["ae_error"] = errs
X_aug["ae_anomaly"] = flags

# ---------- RECONSTRUCTION ERROR VISUALIZATION ----------
plt.figure(figsize=(8, 5))
sns.histplot(errs[y == 0], bins=50, color='green', label='Non-Fraud', stat='density')
sns.histplot(errs[y == 1], bins=50, color='red', label='Fraud', stat='density')
plt.axvline(thresh, color='blue', linestyle='--', label=f'Threshold={thresh:.4f}')
plt.title("Autoencoder Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error")
plt.legend()
plt.tight_layout()
if SAVE_PLOTS: plt.savefig(OUTPUT_DIR / "ae_reconstruction_error.png", dpi=300)
plt.show()

# ---------- RF and XGB Evaluation ----------
X_sup = sup_preproc.transform(X_aug)
rf_probs = rf.predict_proba(X_sup)[:, 1]
xgb_probs = xgb.predict_proba(X_sup)[:, 1]

rf_pred = (rf_probs >= 0.5).astype(int)
xgb_pred = (xgb_probs >= 0.5).astype(int)

# ---------- METRIC CALCULATION ----------
def compute_metrics(y_true, y_pred, y_proba):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_proba)
    }

rf_metrics = compute_metrics(y, rf_pred, rf_probs)
xgb_metrics = compute_metrics(y, xgb_pred, xgb_probs)

metrics_df = pd.DataFrame([rf_metrics, xgb_metrics], index=['RF+AE', 'XGB+AE'])
print("\n=== PERFORMANCE METRICS ===")
print(metrics_df)

# ---------- CONFUSION MATRICES ----------
rf_cm = confusion_matrix(y, rf_pred)
xgb_cm = confusion_matrix(y, xgb_pred)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Confusion Matrix - Random Forest + AE")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(xgb_cm, annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("Confusion Matrix - XGBoost + AE")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
if SAVE_PLOTS: plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=300)
plt.show()

# ---------- ROC CURVES ----------
fpr_rf, tpr_rf, _ = roc_curve(y, rf_probs)
fpr_xgb, tpr_xgb, _ = roc_curve(y, xgb_probs)
roc_auc_rf = auc(fpr_rf, tpr_rf)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

plt.figure(figsize=(6, 6))
plt.plot(fpr_rf, tpr_rf, label=f'RF+AE (AUC = {roc_auc_rf:.3f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGB+AE (AUC = {roc_auc_xgb:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
if SAVE_PLOTS: plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=300)
plt.show()

# ---------- PRECISION-RECALL CURVES ----------
prec_rf, rec_rf, _ = precision_recall_curve(y, rf_probs)
prec_xgb, rec_xgb, _ = precision_recall_curve(y, xgb_probs)

plt.figure(figsize=(6, 6))
plt.plot(rec_rf, prec_rf, label='RF+AE')
plt.plot(rec_xgb, prec_xgb, label='XGB+AE')
plt.title("Precision-Recall Curve Comparison")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid()
if SAVE_PLOTS: plt.savefig(OUTPUT_DIR / "precision_recall_curve.png", dpi=300)
plt.show()

# ---------- FEATURE IMPORTANCE ----------
rf_importance = pd.DataFrame({
    'feature': X_aug.columns,
    'importance': rf.feature_importances_
}).sort_values(by='importance', ascending=False)

xgb_importance = pd.DataFrame({
    'feature': X_aug.columns,
    'importance': xgb.feature_importances_
}).sort_values(by='importance', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(data=rf_importance.head(10), x='importance', y='feature', ax=axes[0], color='skyblue')
axes[0].set_title("Top 10 Features - Random Forest + AE")

sns.barplot(data=xgb_importance.head(10), x='importance', y='feature', ax=axes[1], color='salmon')
axes[1].set_title("Top 10 Features - XGBoost + AE")

plt.tight_layout()
if SAVE_PLOTS: plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300)
plt.show()

print("\nPerformance analysis complete. Plots saved in artifacts folder (if SAVE_PLOTS=True).")
