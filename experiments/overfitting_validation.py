import sys
import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import StratifiedKFold

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

print("============================================================")
print(" STRICT OVERFITTING & GENERALIZATION VALIDATION SUITE")
print("============================================================")

DATA_PATH = "data/cleaned_paysim_lstm.csv"
V3_DIR = "models/paysim_v3"

print("\n[INFO] Loading data and models...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower().str.strip()
df = df.sort_values("step").reset_index(drop=True)

# Engineer velocity features (same as evaluation script)
if "hour" not in df.columns:
    df["hour"] = df["step"] % 24
if "dayofweek" not in df.columns:
    df["dayofweek"] = (df["step"] // 24) % 7
if "is_weekend" not in df.columns:
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
for col in ["upi_type_upi_payment", "upi_type_upi_transfer"]:
    if col in df.columns:
        df[col] = df[col].astype(np.int8)

df["tx_count_cumul"] = df.groupby("nameorig").cumcount() + 1
df["amount_cumul"] = df.groupby("nameorig")["amount"].cumsum()
df["amt_vs_avg"] = df["amount"] / (df["amount_cumul"] / df["tx_count_cumul"] + 1e-6)
df["time_since_last"] = df.groupby("nameorig")["step"].diff().fillna(48).clip(0, 96)
df["amt_to_bal_ratio"] = df["amount"] / (df["oldbalanceorg"] + 1e-6)
df["balance_velocity"] = ((df["newbalanceorig"] - df["oldbalanceorg"]) / (df["amount"] + 1e-6))

df["amount_cumul"] = np.log1p(df["amount_cumul"].clip(0))
df["tx_count_cumul"] = np.log1p(df["tx_count_cumul"])
df["amt_to_bal_ratio"] = np.log1p(df["amt_to_bal_ratio"].clip(0))

n = len(df)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

# Train set and Test set
df_train = df.iloc[:train_end].reset_index(drop=True)
df_test = df.iloc[val_end:].reset_index(drop=True)

y_train = df_train["isfraud"].values.astype(np.int32)
y_test = df_test["isfraud"].values.astype(np.int32)

# Load V3 XGBoost model and features
xgb_model = joblib.load(f"{V3_DIR}/paysim_v3_xgb.pkl")
v3_features = joblib.load(f"{V3_DIR}/paysim_v3_features.pkl")
scaler_v3 = joblib.load(f"{V3_DIR}/paysim_v3_scaler.pkl")
base_features = [f for f in v3_features if f != "ae_recon_error"]

# --- TEST 1: FEATURE IMPORTANCE ANALYSIS ---
print("\n=== TEST 1: FEATURE IMPORTANCE ANALYSIS ===")
importances = xgb_model.feature_importances_
feat_imp = pd.DataFrame({"Feature": v3_features, "Importance": importances})
feat_imp = feat_imp.sort_values(by="Importance", ascending=False)
print("Top 10 Most Important Features:")
print(feat_imp.head(10).to_string(index=False))

# --- TEST 2: OUT-OF-DISTRIBUTION STRESS TEST (DATA LEAKAGE CHECK) ---
print("\n=== TEST 2: OOD STRESS TEST (DATA LEAKAGE) ===")
df_fraud_test = df_test[df_test["isfraud"] == 1].copy()

def get_base_preds(fraud_df, model, scaler, feature_list):
    X = np.zeros((len(fraud_df), len(feature_list)-1), dtype=np.float64)
    for i, feat in enumerate([f for f in feature_list if f != "ae_recon_error"]):
        if feat in fraud_df.columns:
            X[:, i] = fraud_df[feat].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_full = np.column_stack([X, np.zeros(len(fraud_df))])
    X_full_s = scaler.transform(X_full)
    probs = model.predict_proba(X_full_s)[:, 1]
    return probs

probs_orig = get_base_preds(df_fraud_test, xgb_model, scaler_v3, v3_features)
recall_orig = (probs_orig > 0.5).mean()
print(f"Recall on Natural Fraud cases: {recall_orig*100:.2f}%")

df_fraud_test_perturbed = df_fraud_test.copy()
if "errorbalanceorig" in df_fraud_test_perturbed.columns:
    df_fraud_test_perturbed["errorbalanceorig"] = 0.0
if "errorbalancedest" in df_fraud_test_perturbed.columns:
    df_fraud_test_perturbed["errorbalancedest"] = 0.0

probs_pert = get_base_preds(df_fraud_test_perturbed, xgb_model, scaler_v3, v3_features)
recall_pert = (probs_pert > 0.5).mean()
print(f"Recall when 'errorBalanceOrig/Dest' are neutralized: {recall_pert*100:.2f}%")
print("-> If this drops heavily, the model has overfitted to a data leak in the dataset.")

# --- TEST 3: CROSS VALIDATION (STRICT) ---
print("\n=== TEST 3: STRICT CROSS-VALIDATION (WITHOUT LEAKY FEATURES) ===")
print("Training a new XGBoost Model on 3 Folds EXCLUDING errorBalanceOrig and errorBalanceDest...")

safe_features = [f for f in base_features if "errorbalance" not in f.lower()]
X_train_safe = df_train[safe_features].fillna(0).values

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_recalls = []
cv_precisions = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_safe, y_train)):
    X_tr, y_tr = X_train_safe[train_idx], y_train[train_idx]
    X_va, y_va = X_train_safe[val_idx], y_train[val_idx]
    
    clf = xgb.XGBClassifier(n_estimators=50, max_depth=6, scale_pos_weight=100, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    
    preds = clf.predict(X_va)
    cv_recalls.append(recall_score(y_va, preds))
    cv_precisions.append(precision_score(y_va, preds))

print(f"CV Recall without leaks: {np.mean(cv_recalls)*100:.2f}% (+/- {np.std(cv_recalls)*100:.2f}%)")
print(f"CV Precision without leaks: {np.mean(cv_precisions)*100:.2f}% (+/- {np.std(cv_precisions)*100:.2f}%)")

# --- TEST 4: LEARNING CURVES (TRAIN VS TEST GAP) ---
print("\n=== TEST 4: LEARNING CURVES (TRAIN VS TEST) ===")
X_train_full = df_train[base_features].fillna(0).values
X_test_full = df_test[base_features].fillna(0).values

clf_curve = xgb.XGBClassifier(n_estimators=100, max_depth=6, scale_pos_weight=50, random_state=42, n_jobs=-1)
eval_set = [(X_train_full, y_train), (X_test_full, y_test)]
clf_curve.fit(X_train_full, y_train, eval_set=eval_set, verbose=False)

results = clf_curve.evals_result()
train_logloss = results['validation_0']['logloss']
test_logloss = results['validation_1']['logloss']

print(f"Initial Train Logloss: {train_logloss[0]:.4f} | Test Logloss: {test_logloss[0]:.4f}")
print(f"Final Train Logloss: {train_logloss[-1]:.4f} | Test Logloss: {test_logloss[-1]:.4f}")
if test_logloss[-1] > test_logloss[len(test_logloss)//2]:
    print("-> WARNING: Validation loss is increasing while training loss decreases (Classic Overfitting).")
else:
    print("-> Validation loss is stable, but check the absolute values. If they are near 0.000, the dataset is trivially separated.")

print("\n============================================================")
print(" ANALYSIS COMPLETE")
print("============================================================")
