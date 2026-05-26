"""
retrain_paysim_v2.py
────────────────────
Retrain PaySim fraud detection models (XGBoost + Random Forest) on the
FULL 6.36M-row cleaned dataset with SMOTE and algorithmic threshold.

All new models are saved to  models/paysim_v2/  — existing models are NOT touched.

Steps:
  1. Load full cleaned_paysim.csv (6.36M rows)
  2. Remove has_balance_mismatch (data leakage)
  3. Stratified 70/15/15 split
  4. SMOTE on training set only (→ ~10% fraud ratio)
  5. StandardScaler normalization
  6. Train XGBoost with regularization
  7. Train Random Forest with class balancing
  8. Derive threshold via F-beta (β=2) on validation set
  9. Evaluate on held-out test set
 10. Save all artifacts + publication-quality charts

Usage:
    python retrain_paysim_v2.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "cleaned_paysim.csv"
OUTPUT_DIR = BASE_DIR / "models" / "paysim_v2"
CHARTS_DIR = BASE_DIR / "evaluation_results" / "paysim_evaluation_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================

# 12 features — has_balance_mismatch REMOVED (data leakage)
BASE_FEATURES = [
    "amount", "oldbalanceorg", "newbalanceorig",
    "oldbalancedest", "newbalancedest",
    "hour", "dayofweek", "is_weekend",
    "errorbalanceorig", "errorbalancedest",
    "upi_type_upi_payment", "upi_type_upi_transfer",
]

# SMOTE target: boost fraud to ~10% of training data
SMOTE_RATIO = 0.10

# F-beta: β=2 means recall is 2× more important than precision
BETA = 2


def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def print_step(step_num, title, description):
    """Print a step with explanation."""
    print(f"\n{'─'*70}")
    print(f"  STEP {step_num}: {title}")
    print(f"  📝 {description}")
    print(f"{'─'*70}")


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
def load_data():
    print_step(1, "LOADING FULL DATASET",
        "Loading all 6.36M rows from cleaned_paysim.csv.\n"
        "     WHY: More data = model sees more real-world patterns = better generalization.\n"
        "     The current model was trained on only 6,800 rows (0.1% of the data).")

    df = pd.read_csv(DATA_PATH)

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()

    print(f"\n  📂 Loaded: {len(df):,} transactions")
    print(f"  💰 Frauds: {df['isfraud'].sum():,} ({df['isfraud'].mean()*100:.3f}%)")
    print(f"  ✅ Legitimate: {(df['isfraud']==0).sum():,} ({(df['isfraud']==0).mean()*100:.3f}%)")
    print(f"  📊 Columns: {len(df.columns)}")

    return df


# ============================================================
# STEP 2: FEATURE SELECTION
# ============================================================
def select_features(df):
    print_step(2, "FEATURE SELECTION — REMOVING DATA LEAKAGE",
        "Removing 'has_balance_mismatch' feature.\n"
        "     WHY: This feature is derived FROM the fraud label — it tells the model\n"
        "     the answer instead of letting it learn. Using it is 'cheating'.\n"
        "     We keep 12 clean features that represent genuine transaction attributes.")

    # Ensure all features exist
    for col in BASE_FEATURES:
        if col not in df.columns:
            print(f"  ⚠️  Missing column '{col}', creating with zeros")
            df[col] = 0.0

    X = df[BASE_FEATURES].astype(float)
    y = df["isfraud"].astype(int)

    print(f"\n  ✅ Selected {len(BASE_FEATURES)} features:")
    for i, f in enumerate(BASE_FEATURES, 1):
        print(f"     {i:2d}. {f}")

    print(f"\n  ❌ Excluded: has_balance_mismatch (data leakage)")

    return X, y


# ============================================================
# STEP 3: SPLIT DATA
# ============================================================
def split_data(X, y):
    print_step(3, "STRATIFIED DATA SPLIT (70/15/15)",
        "Splitting data into Train (70%), Validation (15%), and Test (15%).\n"
        "     WHY: Train = model learns from this.\n"
        "          Validation = used to tune the threshold (model never trains on this).\n"
        "          Test = final unbiased evaluation (never seen during training OR tuning).\n"
        "     'Stratified' means each split has the same fraud % as the original data.")

    # First split: 85% temp, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Second split: 70% train, 15% validation (from the 85% temp)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
        # 0.1765 of 85% ≈ 15% of total
    )

    splits = {
        "Train": (X_train, y_train),
        "Validation": (X_val, y_val),
        "Test": (X_test, y_test),
    }

    print(f"\n  📊 Split results:")
    for name, (X_s, y_s) in splits.items():
        fraud_n = y_s.sum()
        fraud_pct = y_s.mean() * 100
        print(f"     {name:12s}: {len(X_s):>10,} rows  |  {fraud_n:>5,} frauds ({fraud_pct:.3f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# STEP 4: SMOTE (Training Data Only)
# ============================================================
def apply_smote(X_train, y_train):
    print_step(4, "SMOTE — BALANCING TRAINING DATA",
        f"Using SMOTE to boost fraud ratio to ~{SMOTE_RATIO*100:.0f}% in training set ONLY.\n"
        "     WHY: With only 0.13% fraud, the model is biased toward 'legitimate'.\n"
        "     SMOTE creates synthetic fraud examples by interpolating between existing\n"
        "     fraud samples — giving the model more examples to learn fraud patterns.\n"
        "     ⚠️  CRITICAL: Validation and Test sets are NOT touched — they stay 100% real.")

    before_legit = (y_train == 0).sum()
    before_fraud = (y_train == 1).sum()

    print(f"\n  BEFORE SMOTE:")
    print(f"     Legitimate: {before_legit:>10,}")
    print(f"     Fraud:      {before_fraud:>10,} ({y_train.mean()*100:.3f}%)")

    smote = SMOTE(
        sampling_strategy=SMOTE_RATIO,  # target: fraud = 10% of legit
        random_state=42,
        k_neighbors=5,
    )

    t0 = time.time()
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    elapsed = time.time() - t0

    after_legit = (y_resampled == 0).sum()
    after_fraud = (y_resampled == 1).sum()
    synthetic_count = after_fraud - before_fraud

    print(f"\n  AFTER SMOTE ({elapsed:.1f}s):")
    print(f"     Legitimate: {after_legit:>10,} (unchanged)")
    print(f"     Fraud:      {after_fraud:>10,} ({y_resampled.mean()*100:.2f}%)")
    print(f"     Synthetic:  {synthetic_count:>10,} new fraud samples created")
    print(f"     Total:      {len(X_resampled):>10,} rows")

    return X_resampled, y_resampled, {
        "before_legit": before_legit, "before_fraud": before_fraud,
        "after_legit": after_legit, "after_fraud": after_fraud,
        "synthetic": synthetic_count,
    }


# ============================================================
# STEP 5: SCALE FEATURES
# ============================================================
def scale_features(X_train, X_val, X_test):
    print_step(5, "FEATURE SCALING (StandardScaler)",
        "Normalizing all features to mean=0, std=1.\n"
        "     WHY: Ensures all features are on the same scale. Without scaling,\n"
        "     'amount' (range: 0-90M) would dominate 'is_weekend' (range: 0-1).\n"
        "     The scaler is fit ONLY on training data, then applied to val/test.")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)  # FIT + transform on train
    X_val_s = scaler.transform(X_val)            # Only transform on val
    X_test_s = scaler.transform(X_test)          # Only transform on test

    print(f"\n  ✅ Scaler fitted on training data ({X_train_s.shape[0]:,} rows)")
    print(f"  ✅ Applied to validation ({X_val_s.shape[0]:,} rows)")
    print(f"  ✅ Applied to test ({X_test_s.shape[0]:,} rows)")
    print(f"  ⚠️  Scaler was NOT fitted on val/test (prevents data leakage)")

    return scaler, X_train_s, X_val_s, X_test_s


# ============================================================
# STEP 6: TRAIN XGBOOST
# ============================================================
def train_xgboost(X_train_s, y_train):
    print_step(6, "TRAINING XGBOOST (Gradient Boosting)",
        "Training XGBoost with regularization to prevent overfitting.\n"
        "     HOW: XGBoost builds trees sequentially — each tree fixes the errors\n"
        "     of the previous one. This 'boosting' makes it very accurate.\n"
        "     We add regularization (gamma, lambda, alpha) to prevent the model\n"
        "     from memorizing noise in the data.")

    # Calculate class weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos if pos > 0 else 1

    print(f"\n  ⚙️  Hyperparameters:")
    print(f"     n_estimators   = 300   (number of trees)")
    print(f"     max_depth      = 6     (tree depth limit)")
    print(f"     learning_rate  = 0.05  (step size — smaller = more careful)")
    print(f"     subsample      = 0.8   (use 80% of data per tree — reduces overfitting)")
    print(f"     colsample      = 0.8   (use 80% of features per tree)")
    print(f"     min_child_wt   = 5     (minimum samples per leaf)")
    print(f"     gamma          = 0.1   (regularization — prunes weak splits)")
    print(f"     reg_alpha      = 0.1   (L1 regularization)")
    print(f"     reg_lambda     = 1.0   (L2 regularization)")
    print(f"     scale_pos_wt   = {spw:.2f} (compensates for class imbalance)")

    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=spw,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    xgb.fit(X_train_s, y_train)
    elapsed = time.time() - t0

    print(f"\n  ✅ XGBoost trained in {elapsed:.1f}s")
    print(f"     Trees: {xgb.n_estimators}")
    print(f"     Features used: {xgb.n_features_in_}")

    return xgb


# ============================================================
# STEP 7: TRAIN RANDOM FOREST
# ============================================================
def train_random_forest(X_train_s, y_train):
    print_step(7, "TRAINING RANDOM FOREST (Bagging Ensemble)",
        "Training Random Forest — a different kind of ensemble from XGBoost.\n"
        "     HOW: RF builds 300 independent trees on random subsets of data,\n"
        "     then averages their votes. This 'bagging' reduces variance.\n"
        "     WHY both XGB and RF: They make DIFFERENT errors, so combining them\n"
        "     catches more fraud than either alone.")

    print(f"\n  ⚙️  Hyperparameters:")
    print(f"     n_estimators    = 300     (number of independent trees)")
    print(f"     max_depth       = 12      (tree depth limit)")
    print(f"     min_samples_spl = 10      (minimum samples to split a node)")
    print(f"     min_samples_lf  = 5       (minimum samples per leaf)")
    print(f"     class_weight    = balanced (auto-adjusts for imbalance)")

    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    rf.fit(X_train_s, y_train)
    elapsed = time.time() - t0

    print(f"\n  ✅ Random Forest trained in {elapsed:.1f}s")
    print(f"     Trees: {rf.n_estimators}")
    print(f"     Features used: {rf.n_features_in_}")

    return rf


# ============================================================
# STEP 8: DERIVE THRESHOLD USING F-BETA
# ============================================================
def derive_threshold(xgb, rf, X_val_s, y_val):
    print_step(8, "ALGORITHMIC THRESHOLD DERIVATION (F-beta, β=2)",
        "Finding the optimal decision threshold using the F2 score.\n"
        "     WHY F2: F1 treats precision and recall equally. F2 treats RECALL\n"
        "     as 2× more important — perfect for fraud detection where missing\n"
        "     a fraud (FN) is worse than a false alarm (FP).\n"
        "     HOW: We sweep all possible thresholds (0.01 to 0.99) on the\n"
        "     VALIDATION set and pick the one with the highest F2 score.\n"
        "     ⚠️  This is done on validation data, NOT test data.")

    # Get ensemble probabilities on validation set
    xgb_probs = xgb.predict_proba(X_val_s)[:, 1]
    rf_probs = rf.predict_proba(X_val_s)[:, 1]

    # Test different ensemble weights
    weight_configs = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
    best_overall_f2 = 0
    best_weights = (0.6, 0.4)
    best_threshold = 0.35
    best_metrics = {}

    print(f"\n  🔍 Testing weight combinations:")
    print(f"     {'Weights':15s} | {'Threshold':10s} | {'F2':8s} | {'Recall':8s} | {'Precision':10s}")
    print(f"     {'─'*60}")

    for w_xgb, w_rf in weight_configs:
        ens_probs = w_xgb * xgb_probs + w_rf * rf_probs

        # Sweep thresholds
        thresholds = np.arange(0.01, 0.99, 0.01)
        best_f2_for_weight = 0
        best_t_for_weight = 0.5

        for t in thresholds:
            preds = (ens_probs >= t).astype(int)
            if preds.sum() == 0:
                continue
            f2 = fbeta_score(y_val, preds, beta=BETA, zero_division=0)
            if f2 > best_f2_for_weight:
                best_f2_for_weight = f2
                best_t_for_weight = t

        # Evaluate at best threshold
        preds = (ens_probs >= best_t_for_weight).astype(int)
        rec = recall_score(y_val, preds, zero_division=0)
        prec = precision_score(y_val, preds, zero_division=0)

        print(f"     {w_xgb:.1f} XGB + {w_rf:.1f} RF | {best_t_for_weight:.4f}     | {best_f2_for_weight:.4f} | {rec:.4f}  | {prec:.4f}")

        if best_f2_for_weight > best_overall_f2:
            best_overall_f2 = best_f2_for_weight
            best_weights = (w_xgb, w_rf)
            best_threshold = best_t_for_weight
            best_metrics = {"f2": best_f2_for_weight, "recall": rec, "precision": prec}

    print(f"\n  🏆 Best Configuration:")
    print(f"     Weights:   {best_weights[0]:.1f} XGB + {best_weights[1]:.1f} RF")
    print(f"     Threshold: {best_threshold:.4f}")
    print(f"     F2-Score:  {best_metrics['f2']:.4f}")
    print(f"     Recall:    {best_metrics['recall']:.4f}")
    print(f"     Precision: {best_metrics['precision']:.4f}")

    return best_weights, best_threshold


# ============================================================
# STEP 9: EVALUATE ON TEST SET
# ============================================================
def evaluate_on_test(xgb, rf, X_train_s, y_train, X_test_s, y_test, weights, threshold):
    print_step(9, "FINAL EVALUATION ON HELD-OUT TEST SET",
        "Evaluating on the 15% test set that was NEVER seen during training\n"
        "     or threshold tuning. This gives us an honest estimate of how\n"
        "     the model will perform on brand-new, unseen transactions.")

    w_xgb, w_rf = weights

    # Individual model probabilities
    xgb_probs = xgb.predict_proba(X_test_s)[:, 1]
    rf_probs = rf.predict_proba(X_test_s)[:, 1]
    ens_probs = w_xgb * xgb_probs + w_rf * rf_probs

    # Also get training metrics (to check for overfitting)
    xgb_train_probs = xgb.predict_proba(X_train_s)[:, 1]
    rf_train_probs = rf.predict_proba(X_train_s)[:, 1]
    ens_train_probs = w_xgb * xgb_train_probs + w_rf * rf_train_probs
    ens_train_preds = (ens_train_probs >= threshold).astype(int)
    train_recall = recall_score(y_train, ens_train_preds, zero_division=0)
    train_prec = precision_score(y_train, ens_train_preds, zero_division=0)

    # Evaluate each model
    results = {}

    for name, probs in [("XGBoost", xgb_probs), ("Random Forest", rf_probs), ("Ensemble", ens_probs)]:
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "ROC-AUC": roc_auc_score(y_test, probs),
            "PR-AUC": average_precision_score(y_test, probs),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0),
            "F2": fbeta_score(y_test, preds, beta=2, zero_division=0),
            "Accuracy": accuracy_score(y_test, preds),
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        }
        results[name] = metrics

        print(f"\n  📊 {name}:")
        print(f"     ROC-AUC    : {metrics['ROC-AUC']:.4f}")
        print(f"     PR-AUC     : {metrics['PR-AUC']:.4f}")
        print(f"     Recall     : {metrics['Recall']:.4f} ({metrics['Recall']*100:.2f}%)")
        print(f"     Precision  : {metrics['Precision']:.4f} ({metrics['Precision']*100:.2f}%)")
        print(f"     F1-Score   : {metrics['F1']:.4f}")
        print(f"     F2-Score   : {metrics['F2']:.4f}")
        print(f"     Specificity: {metrics['Specificity']:.4f}")
        print(f"     Confusion  : TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")

    # Overfitting check
    print(f"\n  🔍 OVERFITTING CHECK (Ensemble):")
    print(f"     Training Recall:    {train_recall:.4f}")
    print(f"     Test Recall:        {results['Ensemble']['Recall']:.4f}")
    print(f"     Gap:                {abs(train_recall - results['Ensemble']['Recall']):.4f}")
    gap = abs(train_recall - results["Ensemble"]["Recall"])
    if gap < 0.05:
        print(f"     ✅ Gap < 5% — No overfitting detected!")
    elif gap < 0.10:
        print(f"     ⚠️  Gap 5-10% — Mild overfitting, acceptable")
    else:
        print(f"     ❌ Gap > 10% — Potential overfitting!")

    return results, {
        "xgb_probs": xgb_probs, "rf_probs": rf_probs, "ens_probs": ens_probs,
    }


# ============================================================
# STEP 10: SAVE MODELS + CHARTS
# ============================================================
def save_artifacts(scaler, xgb, rf, threshold, weights):
    print_step("10a", "SAVING MODEL ARTIFACTS",
        f"Saving all trained models to  models/paysim_v2/\n"
        f"     ⚠️  Existing models in models/ are NOT touched.")

    joblib.dump(xgb, OUTPUT_DIR / "paysim_xgb_stage2.pkl")
    joblib.dump(rf, OUTPUT_DIR / "paysim_rf_stage2.pkl")
    joblib.dump(scaler, OUTPUT_DIR / "paysim_stage2_scaler.pkl")
    joblib.dump(BASE_FEATURES, OUTPUT_DIR / "paysim_stage2_features.pkl")
    np.save(OUTPUT_DIR / "paysim_stage2_threshold.npy", np.array([threshold]))
    np.save(OUTPUT_DIR / "paysim_stage2_weights.npy", np.array(weights))

    print(f"\n  ✅ paysim_xgb_stage2.pkl")
    print(f"  ✅ paysim_rf_stage2.pkl")
    print(f"  ✅ paysim_stage2_scaler.pkl")
    print(f"  ✅ paysim_stage2_features.pkl (12 features)")
    print(f"  ✅ paysim_stage2_threshold.npy ({threshold:.4f})")
    print(f"  ✅ paysim_stage2_weights.npy ({weights[0]:.1f}/{weights[1]:.1f})")


def save_charts(smote_stats, y_train_before, y_val, y_test, y_train_after,
                results, xgb, rf, probs_dict, y_test_arr, weights, threshold):
    print_step("10b", "GENERATING PUBLICATION-QUALITY CHARTS",
        "Creating histogram and comparison charts for your report/paper.")

    # ─── Chart 1: Data Split Distribution ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PaySim Dataset — Train/Validation/Test Split", fontsize=16, fontweight="bold")

    # Bar chart of split sizes
    ax = axes[0]
    splits = ["Train\n(70%)", "Validation\n(15%)", "Test\n(15%)"]
    sizes = [len(y_train_before), len(y_val), len(y_test)]
    bars = ax.bar(splits, sizes, color=["#2196F3", "#FF9800", "#4CAF50"],
                  edgecolor="white", linewidth=2)
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30000,
                f"{size:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Transactions", fontsize=12)
    ax.set_title("Split Sizes", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))

    # Fraud % per split
    ax = axes[1]
    fraud_pcts = [
        y_train_before.mean() * 100,
        y_val.mean() * 100,
        y_test.mean() * 100,
    ]
    bars = ax.bar(splits, fraud_pcts, color=["#2196F3", "#FF9800", "#4CAF50"],
                  edgecolor="white", linewidth=2)
    for bar, pct in zip(bars, fraud_pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{pct:.3f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Fraud Percentage (%)", fontsize=12)
    ax.set_title("Fraud Rate per Split (Stratified)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path1 = CHARTS_DIR / "paysim_data_split.png"
    fig.savefig(path1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path1}")

    # ─── Chart 2: Before vs After SMOTE ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PaySim Training Data — Before vs After SMOTE", fontsize=16, fontweight="bold")

    # Before SMOTE
    ax = axes[0]
    labels = ["Legitimate", "Fraud"]
    before_vals = [smote_stats["before_legit"], smote_stats["before_fraud"]]
    bars = ax.bar(labels, before_vals, color=["#4CAF50", "#F44336"], edgecolor="white", linewidth=2)
    for bar, val in zip(bars, before_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000,
                f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("BEFORE SMOTE", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    pct_before = smote_stats["before_fraud"] / (smote_stats["before_legit"] + smote_stats["before_fraud"]) * 100
    ax.text(0.5, 0.95, f"Fraud: {pct_before:.3f}%", transform=ax.transAxes,
            ha="center", fontsize=12, color="#F44336", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#F44336"))

    # After SMOTE
    ax = axes[1]
    after_vals = [smote_stats["after_legit"], smote_stats["after_fraud"]]
    bars = ax.bar(labels, after_vals, color=["#4CAF50", "#F44336"], edgecolor="white", linewidth=2)
    # Stacked: show real vs synthetic in fraud bar
    ax.bar(["Fraud"], [smote_stats["before_fraud"]], color="#E57373", edgecolor="white", linewidth=2,
           label="Real Fraud")
    ax.bar(["Fraud"], [smote_stats["synthetic"]], bottom=[smote_stats["before_fraud"]],
           color="#F44336", edgecolor="white", linewidth=2, label="Synthetic (SMOTE)")
    for bar, val in zip(bars, after_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000,
                f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("AFTER SMOTE (Training Only)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    pct_after = smote_stats["after_fraud"] / (smote_stats["after_legit"] + smote_stats["after_fraud"]) * 100
    ax.text(0.5, 0.95, f"Fraud: {pct_after:.2f}%", transform=ax.transAxes,
            ha="center", fontsize=12, color="#F44336", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#F44336"))
    ax.legend(fontsize=10)

    plt.tight_layout()
    path2 = CHARTS_DIR / "paysim_smote_comparison.png"
    fig.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path2}")

    # ─── Chart 3: Confusion Matrices ───
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("PaySim v2 Models — Confusion Matrices", fontsize=16, fontweight="bold", y=1.02)

    for idx, name in enumerate(["XGBoost", "Random Forest", "Ensemble"]):
        ax = axes[idx]
        r = results[name]
        cm_display = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        im = ax.imshow(cm_display, cmap="Blues", aspect="auto")

        labels_cm = [[f"TN\n{r['TN']:,}", f"FP\n{r['FP']:,}"],
                      [f"FN\n{r['FN']:,}", f"TP\n{r['TP']:,}"]]
        for i in range(2):
            for j in range(2):
                color = "white" if cm_display[i, j] > cm_display.max() / 2 else "black"
                ax.text(j, i, labels_cm[i][j], ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Legit", "Fraud"], fontsize=11)
        ax.set_yticklabels(["Legit", "Fraud"], fontsize=11)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"{name}", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path3 = CHARTS_DIR / "paysim_v2_confusion_matrices.png"
    fig.savefig(path3, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path3}")

    # ─── Chart 4: ROC Curves ───
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    for idx, (name, probs) in enumerate([
        ("XGBoost", probs_dict["xgb_probs"]),
        ("Random Forest", probs_dict["rf_probs"]),
        ("Ensemble", probs_dict["ens_probs"]),
    ]):
        fpr, tpr, _ = roc_curve(y_test_arr, probs)
        auc_val = roc_auc_score(y_test_arr, probs)
        ax.plot(fpr, tpr, color=colors[idx], linewidth=2.5,
                label=f"{name} (AUC={auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("PaySim v2 — ROC Curves", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path4 = CHARTS_DIR / "paysim_v2_roc_curves.png"
    fig.savefig(path4, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path4}")

    # ─── Chart 5: PR Curves ───
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (name, probs) in enumerate([
        ("XGBoost", probs_dict["xgb_probs"]),
        ("Random Forest", probs_dict["rf_probs"]),
        ("Ensemble", probs_dict["ens_probs"]),
    ]):
        prec_arr, rec_arr, _ = precision_recall_curve(y_test_arr, probs)
        pr_auc_val = average_precision_score(y_test_arr, probs)
        ax.plot(rec_arr, prec_arr, color=colors[idx], linewidth=2.5,
                label=f"{name} (PR-AUC={pr_auc_val:.4f})")

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("PaySim v2 — Precision-Recall Curves", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path5 = CHARTS_DIR / "paysim_v2_pr_curves.png"
    fig.savefig(path5, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path5}")

    # ─── Chart 6: Metrics Bar Chart ───
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_keys = ["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1", "F2"]
    x = np.arange(len(metrics_keys))
    bar_w = 0.25

    for idx, name in enumerate(["XGBoost", "Random Forest", "Ensemble"]):
        vals = [results[name][k] for k in metrics_keys]
        bars = ax.bar(x + idx * bar_w, vals, bar_w, label=name,
                      color=colors[idx], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x + bar_w)
    ax.set_xticklabels(metrics_keys, fontsize=11)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("PaySim v2 — Key Performance Metrics", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path6 = CHARTS_DIR / "paysim_v2_metrics_bar.png"
    fig.savefig(path6, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path6}")

    # ─── Save Markdown comparison ───
    md_path = CHARTS_DIR / "paysim_v2_evaluation.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PaySim v2 — Model Evaluation Results\n\n")
        f.write(f"> **Trained on**: Full cleaned_paysim.csv ({smote_stats['before_legit'] + smote_stats['before_fraud']:,} rows)\n")
        f.write(f"> **SMOTE**: Fraud boosted from {pct_before:.3f}% to {pct_after:.2f}% (training only)\n")
        f.write(f"> **Threshold**: {threshold:.4f} (derived via F2-score)\n")
        f.write(f"> **Weights**: {weights[0]:.1f} XGB + {weights[1]:.1f} RF\n\n")

        f.write("## Performance Metrics (Test Set)\n\n")
        f.write("| Metric | XGBoost | Random Forest | Ensemble |\n")
        f.write("|:-------|:-------:|:-------------:|:--------:|\n")
        for key in ["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1", "F2", "Accuracy", "Specificity"]:
            vals = [f"{results[n][key]:.4f}" for n in ["XGBoost", "Random Forest", "Ensemble"]]
            f.write(f"| **{key}** | {vals[0]} | {vals[1]} | {vals[2]} |\n")

        f.write("\n## Confusion Matrices (Test Set)\n\n")
        for name in ["XGBoost", "Random Forest", "Ensemble"]:
            r = results[name]
            f.write(f"### {name}\n\n```\n")
            f.write(f"                 Predicted\n")
            f.write(f"               Legit    Fraud\n")
            f.write(f"Actual Legit   {r['TN']:>8,}   {r['FP']:>5,}\n")
            f.write(f"Actual Fraud   {r['FN']:>8,}   {r['TP']:>5,}\n")
            f.write(f"```\n\n")

    print(f"  ✅ {md_path}")

    # Save CSV
    csv_path = CHARTS_DIR / "paysim_v2_evaluation.csv"
    rows = []
    for name in ["XGBoost", "Random Forest", "Ensemble"]:
        row = {"Model": name, "Threshold": threshold}
        row.update(results[name])
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  ✅ {csv_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print_header("PAYSIM MODEL RETRAINING — V2")
    print(f"  All new models will be saved to: {OUTPUT_DIR}")
    print(f"  Charts will be saved to: {CHARTS_DIR}")
    print(f"  ⚠️  Existing models in models/ will NOT be modified")

    total_start = time.time()

    # Step 1: Load
    df = load_data()

    # Step 2: Features
    X, y = select_features(df)

    # Step 3: Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Store pre-SMOTE train labels for charts
    y_train_before = y_train.copy()

    # Step 4: SMOTE
    X_train_smote, y_train_smote, smote_stats = apply_smote(X_train, y_train)

    # Step 5: Scale
    scaler, X_train_s, X_val_s, X_test_s = scale_features(X_train_smote, X_val, X_test)

    # Step 6: Train XGBoost
    xgb = train_xgboost(X_train_s, y_train_smote)

    # Step 7: Train Random Forest
    rf = train_random_forest(X_train_s, y_train_smote)

    # Step 8: Derive threshold
    weights, threshold = derive_threshold(xgb, rf, X_val_s, y_val)

    # Step 9: Evaluate
    results, probs_dict = evaluate_on_test(
        xgb, rf, X_train_s, y_train_smote, X_test_s, y_test, weights, threshold
    )

    # Step 10a: Save models
    save_artifacts(scaler, xgb, rf, threshold, weights)

    # Step 10b: Save charts
    save_charts(
        smote_stats, y_train_before, y_val, y_test, y_train_smote,
        results, xgb, rf, probs_dict, y_test.values, weights, threshold,
    )

    total_elapsed = time.time() - total_start

    print_header("🎉 RETRAINING COMPLETE!")
    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"\n  📦 Models saved to:  {OUTPUT_DIR}")
    print(f"  📊 Charts saved to:  {CHARTS_DIR}")
    print(f"\n  Files created:")
    print(f"     Models:")
    print(f"       • paysim_xgb_stage2.pkl")
    print(f"       • paysim_rf_stage2.pkl")
    print(f"       • paysim_stage2_scaler.pkl")
    print(f"       • paysim_stage2_features.pkl")
    print(f"       • paysim_stage2_threshold.npy")
    print(f"       • paysim_stage2_weights.npy")
    print(f"     Charts:")
    print(f"       • paysim_data_split.png")
    print(f"       • paysim_smote_comparison.png")
    print(f"       • paysim_v2_confusion_matrices.png")
    print(f"       • paysim_v2_roc_curves.png")
    print(f"       • paysim_v2_pr_curves.png")
    print(f"       • paysim_v2_metrics_bar.png")
    print(f"       • paysim_v2_evaluation.md")
    print(f"       • paysim_v2_evaluation.csv")
    print(f"\n  Next steps:")
    print(f"     1. Compare v2 metrics vs current v1 metrics")
    print(f"     2. If v2 is better, copy models from paysim_v2/ to models/")
    print(f"     3. Restart the backend to load new models")


if __name__ == "__main__":
    main()
