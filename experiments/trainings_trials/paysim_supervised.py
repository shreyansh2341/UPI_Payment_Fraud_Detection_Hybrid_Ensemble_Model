import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                             precision_recall_curve, auc, precision_score, recall_score, f1_score)

LAPTOP_TEST = True          
SAMPLE_N = 10000            
DATA_DIR = Path("data")     
PAYSIM_FILE = DATA_DIR / "cleaned_paysim.csv"
CREDIT_FILE = DATA_DIR / "cleaned_creditcard.csv"
OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
DATASET = "paysim"         
RF_N_ESTIMATORS = 200
XGB_N_ESTIMATORS = 200
# ----------------------------------------

def load_dataset(dataset_name="paysim", laptop_mode=True, sample_n=10000):
    if dataset_name == "paysim":
        path = PAYSIM_FILE
    elif dataset_name == "creditcard":
        path = CREDIT_FILE
    else:
        raise ValueError("dataset_name must be 'paysim' or 'creditcard'")
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    print("Full shape:", df.shape)
    if laptop_mode:
        n = min(sample_n, len(df))
        df = df.sample(n, random_state=RANDOM_STATE).reset_index(drop=True)
        print("Sampled shape for laptop test:", df.shape)
    return df

def basic_eda(df):
    print("Columns:", df.columns.tolist())
    print("Target distribution:\n", df['isFraud'].value_counts(), "\nFraud fraction:", df['isFraud'].mean())

def prepare_features(df):
    y = df['isFraud'].astype(int)
    X = df.drop(columns=['isFraud']).copy()
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print("Warning: non-numeric columns found and will be dropped:", non_numeric)
        X = X.drop(columns=non_numeric)
    return X, y

def train_test_split_strat(X, y, test_size=0.2):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    print("Train shape:", X_tr.shape, "Test shape:", X_te.shape)
    print("Train fraud fraction:", y_tr.mean(), "Test fraud fraction:", y_te.mean())
    return X_tr, X_te, y_tr, y_te

def build_preprocessor():
    """Simple numeric preprocessor: median impute + robust scaler (resistant to outliers)."""
    preproc = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    return preproc

def train_models(X_tr_t, y_tr):
    """Train RandomForest and XGBoost on preprocessed data arrays."""
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_tr_t, y_tr)
    pos = y_tr.sum()
    neg = len(y_tr) - pos
    scale_pos_weight = (neg / (pos + 1e-9))
    xgb = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    xgb.fit(X_tr_t, y_tr)
    print(f"Trained RF and XGB. XGB scale_pos_weight={scale_pos_weight:.2f}")
    return rf, xgb

def evaluate_model(model, X_te_t, y_te, model_name="model", plot_curves=True):
    proba = model.predict_proba(X_te_t)[:,1]
    pred = (proba >= 0.5).astype(int)

    print(f"\n=== Evaluation for {model_name} (threshold=0.5) ===")
    print("Confusion matrix:\n", confusion_matrix(y_te, pred))
    print("\nClassification report:\n", classification_report(y_te, pred, digits=4))

    roc = roc_auc_score(y_te, proba)
    precision, recall, thresholds = precision_recall_curve(y_te, proba)
    pr_auc = auc(recall, precision)
    print(f"ROC-AUC: {roc:.4f}    PR-AUC: {pr_auc:.4f}")

    if plot_curves:
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_te, proba)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={roc:.3f})')
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        # PR curve
        plt.subplot(1,2,2)
        plt.plot(recall, precision, label=f'{model_name} (PR-AUC={pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.tight_layout()
        plt.show()

    results = {
        'roc_auc': roc,
        'pr_auc': pr_auc,
        'precision_at_0.5': precision_score(y_te, pred),
        'recall_at_0.5': recall_score(y_te, pred),
        'f1_at_0.5': f1_score(y_te, pred)
    }
    return results, proba

def feature_importance_report(model, feature_names, top_n=15, model_name="model"):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        try:
            importances = model.feature_importances_
        except Exception as e:
            print("Feature importances not available for this model:", e)
            return
    fi = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi = fi.sort_values('importance', ascending=False).reset_index(drop=True)
    print(f"\nTop {top_n} features for {model_name}:")
    print(fi.head(top_n))
    # quick barplot
    plt.figure(figsize=(8,5))
    sns.barplot(data=fi.head(top_n), x='importance', y='feature')
    plt.title(f"Top {top_n} feature importances ({model_name})")
    plt.tight_layout()
    plt.show()
    return fi

def compare_models(metrics_rf, metrics_xgb):
    comp = pd.DataFrame({
        'model': ['RandomForest','XGBoost'],
        'roc_auc': [metrics_rf['roc_auc'], metrics_xgb['roc_auc']],
        'pr_auc': [metrics_rf['pr_auc'], metrics_xgb['pr_auc']],
        'precision@0.5': [metrics_rf['precision_at_0.5'], metrics_xgb['precision_at_0.5']],
        'recall@0.5': [metrics_rf['recall_at_0.5'], metrics_xgb['recall_at_0.5']],
        'f1@0.5': [metrics_rf['f1_at_0.5'], metrics_xgb['f1_at_0.5']]
    })
    print("\nModel comparison:")
    print(comp)
    return comp

def save_models(rf, xgb, preproc, dataset_name):
    joblib.dump(preproc, OUTPUT_DIR / f"{dataset_name}_preproc.joblib")
    joblib.dump(rf, OUTPUT_DIR / f"{dataset_name}_rf.joblib")
    joblib.dump(xgb, OUTPUT_DIR / f"{dataset_name}_xgb.joblib")
    print("Saved preprocessor and models to", OUTPUT_DIR)

def main():
    df = load_dataset(DATASET, laptop_mode=LAPTOP_TEST, sample_n=SAMPLE_N)
    basic_eda(df)

    X, y = prepare_features(df)

    X_tr, X_te, y_tr, y_te = train_test_split_strat(X, y)

    preproc = build_preprocessor()
    print("Fitting preprocessor on training data...")
    X_tr_t = preproc.fit_transform(X_tr)
    X_te_t = preproc.transform(X_te)

    rf, xgb = train_models(X_tr_t, y_tr)

    metrics_rf, proba_rf = evaluate_model(rf, X_te_t, y_te, model_name="RandomForest")
    metrics_xgb, proba_xgb = evaluate_model(xgb, X_te_t, y_te, model_name="XGBoost")

    feature_names = X_tr.columns.tolist()
    fi_rf = feature_importance_report(rf, feature_names, top_n=12, model_name="RandomForest")
    fi_xgb = feature_importance_report(xgb, feature_names, top_n=12, model_name="XGBoost")

    comp = compare_models(metrics_rf, metrics_xgb)
    save_models(rf, xgb, preproc, DATASET)

    if LAPTOP_TEST:
        print("\nRunning quick 3-fold cross-validation ROC-AUC (RF) on sampled data...")
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(rf, X_tr_t, y_tr, cv=skf, scoring='roc_auc', n_jobs=-1)
        print("RF CV ROC-AUC scores:", scores, "mean:", scores.mean())

if __name__ == "__main__":
    main()
