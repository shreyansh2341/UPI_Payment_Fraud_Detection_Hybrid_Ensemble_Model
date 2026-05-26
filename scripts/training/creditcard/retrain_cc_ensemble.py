"""
Retrain BOTH XGBoost and Random Forest for Credit Card Ensemble
This fixes the feature mismatch bug causing XGBoost to return 0.000
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score

print("="*80)
print(" RETRAINING CREDIT CARD ENSEMBLE (XGBoost + Random Forest)")
print("="*80)

# Load Data
print("\nLoading credit card data...")
df = pd.read_csv("data/cleaned_creditcard.csv")
# Convert column names to lowercase to match features file
df.columns = df.columns.str.lower()
print(f"   Loaded {len(df):,} transactions")
print(f"   Frauds: {df['isfraud'].sum():,} ({df['isfraud'].mean()*100:.2f}%)")

# Split Data
print("\nSplitting data (70/15/15)...")
X = df.drop('isfraud', axis=1)
y = df['isfraud']

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# Load Scaler and Features
print("\nLoading scaler and features...")
scaler = joblib.load("models/cc_scaler.pkl")
features = joblib.load("models/cc_features.pkl")
print(f"   Features: {len(features)}")

# Scale Data
X_train_scaled = scaler.transform(X_train[features])
X_val_scaled = scaler.transform(X_val[features])
X_test_scaled = scaler.transform(X_test[features])

# Calculate Class Weight
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

# TRAIN XGBOOST
print("\nRetraining XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=42,
    n_jobs=-1,
    verbosity=1
)
xgb_model.fit(X_train_scaled, y_train)

xgb_prob_test = xgb_model.predict_proba(X_test_scaled)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_prob_test)
print(f"   XGBoost ROC-AUC: {xgb_auc:.4f}")

# Save XGBoost
joblib.dump(xgb_model, "models/cc_xgb_model.pkl")
print(f"   SAVED: cc_xgb_model.pkl")

# TRAIN RANDOM FOREST
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight={0: 1.0, 1: scale_pos},
    random_state=42,
    n_jobs=-1,
    verbose=1
)
rf_model.fit(X_train_scaled, y_train)

rf_prob_test = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_auc = roc_auc_score(y_test, rf_prob_test)
print(f"\n   Random Forest ROC-AUC: {rf_auc:.4f}")

# Save RF
joblib.dump(rf_model, "models/cc_rf_model.pkl")
print(f"   SAVED: cc_rf_model.pkl")

# OPTIMIZE ENSEMBLE
print("\nOptimizing Ensemble Weights...")

weight_combinations = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]

best_f1 = 0
best_weights = (0.5, 0.5)
best_threshold = 0.5

xgb_prob_val = xgb_model.predict_proba(X_val_scaled)[:, 1]
rf_prob_val = rf_model.predict_proba(X_val_scaled)[:, 1]

for w_xgb, w_rf in weight_combinations:
    ensemble_prob_val = w_xgb * xgb_prob_val + w_rf * rf_prob_val
    
    precision, recall, thresholds = precision_recall_curve(y_val, ensemble_prob_val)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    
    valid_idx = np.where(recall[:-1] >= 0.70)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmax(f1_scores[valid_idx])]
    else:
        best_idx = np.argmax(f1_scores)
    
    threshold = thresholds[best_idx]
    f1 = f1_scores[best_idx]
    prec = precision[best_idx]
    rec = recall[best_idx]
    
    print(f"   {w_xgb:.1f} XGB + {w_rf:.1f} RF | Thr={threshold:.4f} | F1={f1:.4f} | P={prec:.4f} R={rec:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_weights = (w_xgb, w_rf)
        best_threshold = threshold

print(f"\nBest: {best_weights[0]:.1f} XGB + {best_weights[1]:.1f} RF | Threshold={best_threshold:.6f} | F1={best_f1:.4f}")

# Save Ensemble Artifacts
np.save("models/cc_ensemble_weights.npy", np.array(best_weights))
np.save("models/cc_ensemble_threshold.npy", np.array([best_threshold]))

print(f"\nSaved:")
print(f"   cc_ensemble_weights.npy")
print(f"   cc_ensemble_threshold.npy")

# Test Ensemble
ensemble_prob_test = best_weights[0] * xgb_prob_test + best_weights[1] * rf_prob_test
ensemble_pred_test = (ensemble_prob_test >= best_threshold).astype(int)

tp = ((y_test == 1) & (ensemble_pred_test == 1)).sum()
fp = ((y_test == 0) & (ensemble_pred_test == 1)).sum()
tn = ((y_test == 0) & (ensemble_pred_test == 0)).sum()
fn = ((y_test == 1) & (ensemble_pred_test == 1)).sum()

test_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
test_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
test_f1 = 2 * (test_prec * test_rec) / (test_prec + test_rec) if (test_prec + test_rec) > 0 else 0

print(f"\nTest Set Performance:")
print(f"   Precision: {test_prec:.4f} ({test_prec*100:.1f}%)")
print(f"   Recall:    {test_rec:.4f} ({test_rec*100:.1f}%)")
print(f"   F1-Score:  {test_f1:.4f}")
print(f"   TP={tp}, FP={fp}, TN={tn}, FN={fn}")

print("\n" + "="*80)
print(" RETRAINING COMPLETE!")
print("="*80)
print("\nRestart backend to load new models:")
print("  Ctrl+C in backend terminal, then rerun")
print("="*80)
