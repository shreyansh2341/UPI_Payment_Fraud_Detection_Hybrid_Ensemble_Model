"""
train_creditcard_ensemble.py
────────────────────────────
Train Random Forest + Optimize XGBoost+RF Ensemble for Credit Card Fraud Detection

Based on existing XGBoost model, this script:
Train Credit Card Ensemble Model (XGBoost + Random Forest)
UPDATED: Now retrains BOTH XGBoost and Random Forest with matching features
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

print("="*80)
print(" CREDIT CARD ENSEMBLE TRAINING (XGBoost + Random Forest)")
print("="*80)

# --------------------------------------------------
# 1. Load Data
# --------------------------------------------------
print("\n📂 Loading credit card data...")
df = pd.read_csv(DATA_DIR / "cleaned_creditcard.csv")
print(f"   Loaded {len(df):,} transactions")
print(f"   Frauds: {df['isfraud'].sum():,} ({df['isfraud'].mean()*100:.2f}%)")

# --------------------------------------------------
# 2. Split Data
# --------------------------------------------------
print("\n✂️  Splitting data (70/15/15)...")
X = df.drop('isfraud', axis=1)
y = df['isfraud']

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 of 85% ≈ 15% overall
)

print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# --------------------------------------------------
# 3. Load Scaler and Features (from original training)
# --------------------------------------------------
print("\n📊 Loading scaler and features...")
scaler = joblib.load(MODELS_DIR / "cc_scaler.pkl")
features = joblib.load(MODELS_DIR / "cc_features.pkl")

print(f"   Features: {len(features)}")

X = df[features]
y = df[TARGET].values

print(f"   Total samples: {len(df):,}")
print(f"   Fraud ratio: {y.mean():.4f} ({y.sum():,} frauds)")

# Use same 70/15/15 split as XGBoost training
n = len(df)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

X_train, y_train = X.iloc[:train_end], y[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y[val_end:]

print(f"\n📊 Data Split:")
print(f"   Train: {len(X_train):,} ({y_train.sum():,} frauds)")
print(f"   Val:   {len(X_val):,} ({y_val.sum():,} frauds)")
print(f"   Test:  {len(X_test):,} ({y_test.sum():,} frauds)")

# ======================================================
# LOAD EXISTING SCALER & SCALE DATA
# ======================================================
print("\n⚙️  Loading existing scaler...")
scaler = joblib.load(EXISTING_SCALER)

X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ======================================================
# TRAIN RANDOM FOREST
# ======================================================
print("\n🌲 Training Random Forest...")

# Calculate class weight
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
class_weight = {0: 1.0, 1: scale_pos}

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight=class_weight,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train_scaled, y_train)

# Evaluate RF alone
rf_prob_test = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_auc = roc_auc_score(y_test, rf_prob_test)
rf_prauc = average_precision_score(y_test, rf_prob_test)

print(f"\n✅ Random Forest Trained!")
print(f"   ROC-AUC: {rf_auc:.4f}")
print(f"   PR-AUC:  {rf_prauc:.4f}")

# Save RF model
joblib.dump(rf_model, OUTPUT_RF)
print(f"   Saved: {OUTPUT_RF}")

# ======================================================
# LOAD EXISTING XGBOOST
# ======================================================
print("\n📦 Loading existing XGBoost model...")
xgb_model = joblib.load(EXISTING_XGB)

xgb_prob_test = xgb_model.predict_proba(X_test_scaled)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_prob_test)
xgb_prauc = average_precision_score(y_test, xgb_prob_test)

print(f"   XGBoost ROC-AUC: {xgb_auc:.4f}")
print(f"   XGBoost PR-AUC:  {xgb_prauc:.4f}")

# ======================================================
# OPTIMIZE ENSEMBLE WEIGHTS
# ======================================================
print("\n🔧 Optimizing Ensemble Weights...")

# Test weight combinations
weight_combinations = [
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
]

best_f1 = 0
best_weights = (0.6, 0.4)
best_threshold = 0.5
best_metrics = {}

print("\n   Testing weight combinations on validation set:")
print("   " + "-"*70)

for w_xgb, w_rf in weight_combinations:
    # Get probabilities on validation set
    xgb_prob_val = xgb_model.predict_proba(X_val_scaled)[:, 1]
    rf_prob_val = rf_model.predict_proba(X_val_scaled)[:, 1]
    
    # Ensemble probabilities
    ensemble_prob_val = w_xgb * xgb_prob_val + w_rf * rf_prob_val
    
    # Find best threshold for this weight combination
    precision, recall, thresholds = precision_recall_curve(y_val, ensemble_prob_val)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    
    # Find threshold with best F1 (and recall >= 0.70)
    valid_idx = np.where(recall[:-1] >= 0.70)[0]
    
    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmax(f1_scores[valid_idx])]
        threshold = thresholds[best_idx]
        f1 = f1_scores[best_idx]
        prec = precision[best_idx]
        rec = recall[best_idx]
    else:
        # Fallback: max F1 regardless of recall
        best_idx = np.argmax(f1_scores)
        threshold = thresholds[best_idx]
        f1 = f1_scores[best_idx]
        prec = precision[best_idx]
        rec = recall[best_idx]
    
    print(f"   {w_xgb:.1f} XGB + {w_rf:.1f} RF  |  Threshold={threshold:.4f}  |  "
          f"F1={f1:.4f}  |  Prec={prec:.4f}  Rec={rec:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_weights = (w_xgb, w_rf)
        best_threshold = threshold
        best_metrics = {
            'precision': prec,
            'recall': rec,
            'f1': f1
        }

print("   " + "-"*70)
print(f"\n✅ Best Configuration:")
print(f"   Weights: {best_weights[0]:.1f} XGB + {best_weights[1]:.1f} RF")
print(f"   Threshold: {best_threshold:.6f}")
print(f"   Validation F1: {best_metrics['f1']:.4f}")
print(f"   Validation Precision: {best_metrics['precision']:.4f}")
print(f"   Validation Recall: {best_metrics['recall']:.4f}")

# ======================================================
# EVALUATE ON TEST SET
# ======================================================
print("\n📊 Final Evaluation on Test Set...")

# Ensemble probabilities on test set
ensemble_prob_test = best_weights[0] * xgb_prob_test + best_weights[1] * rf_prob_test
ensemble_pred_test = (ensemble_prob_test >= best_threshold).astype(int)

# Metrics
test_auc = roc_auc_score(y_test, ensemble_prob_test)
test_prauc = average_precision_score(y_test, ensemble_prob_test)

cm = confusion_matrix(y_test, ensemble_pred_test)
tn, fp, fn, tp = cm.ravel()

test_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
test_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0

print(f"\n{'='*60}")
print(" ENSEMBLE PERFORMANCE (Test Set)")
print(f"{'='*60}")
print(f"   ROC-AUC: {test_auc:.4f}")
print(f"   PR-AUC:  {test_prauc:.4f}")
print(f"\n   Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
print(f"   Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
print(f"   F1-Score:  {test_f1:.4f}")
print(f"\n   Confusion Matrix:")
print(f"      TN={tn:,}  FP={fp:,}")
print(f"      FN={fn:,}  TP={tp:,}")
print(f"\n   Alert Rate: {(tp + fp) / len(y_test):.2%}")
print(f"{'='*60}")

# ======================================================
# SAVE ENSEMBLE ARTIFACTS
# ======================================================
print("\n💾 Saving ensemble artifacts...")

np.save(OUTPUT_WEIGHTS, np.array(best_weights))
np.save(OUTPUT_THRESHOLD, np.array([best_threshold]))

print(f"   ✅ Weights saved: {OUTPUT_WEIGHTS}")
print(f"   ✅ Threshold saved: {OUTPUT_THRESHOLD}")
print(f"   ✅ RF model saved: {OUTPUT_RF}")

print("\n" + "="*60)
print(" ✨ CREDIT CARD ENSEMBLE TRAINING COMPLETE!")
print("="*60)
print(f"\nNext steps:")
print(f"  1. Update model_loader.py to load RF + weights")
print(f"  2. Update final_ensemble_inference.py for ensemble logic")
print(f"  3. Test with: python test_cc_model.py")
print("="*60)
