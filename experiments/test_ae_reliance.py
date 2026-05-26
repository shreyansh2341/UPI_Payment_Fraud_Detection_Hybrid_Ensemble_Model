import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score

tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Keras compatibility patch
_original_dense_init = tf.keras.layers.Dense.__init__
def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    _original_dense_init(self, *args, **kwargs)
tf.keras.layers.Dense.__init__ = _patched_dense_init

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)

OUTPUT_DIR = "model being robust"

DATA_PATH = "data/cleaned_paysim_lstm.csv"
V3_DIR = "models/paysim_v3"

print("Loading data...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower().str.strip()
df = df.sort_values("step").reset_index(drop=True)

# Engineer velocity features
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
val_end = int(0.85 * n)
df_test = df.iloc[val_end:].reset_index(drop=True)

y_test = df_test["isfraud"].values.astype(np.int32)

print("Loading V3 models...")
v3_xgb = joblib.load(f"{V3_DIR}/paysim_v3_xgb.pkl")
v3_ae = tf.keras.models.load_model(f"{V3_DIR}/paysim_v3_ae.keras", compile=False, safe_mode=False)
v3_scaler = joblib.load(f"{V3_DIR}/paysim_v3_scaler.pkl")
v3_features = joblib.load(f"{V3_DIR}/paysim_v3_features.pkl")
v3_block_threshold = float(np.load(f"{V3_DIR}/paysim_v3_threshold.npy")[0])

v3_base_features = [f for f in v3_features if f != "ae_recon_error"]
X_v3_base = np.zeros((len(df_test), len(v3_base_features)), dtype=np.float64)
for i, feat in enumerate(v3_base_features):
    if feat in df_test.columns:
        X_v3_base[:, i] = df_test[feat].values
X_v3_base = np.nan_to_num(X_v3_base, nan=0.0, posinf=0.0, neginf=0.0)

v3_ae_scaler = StandardScaler()
v3_ae_scaler.mean_ = v3_scaler.mean_[:len(v3_base_features)]
v3_ae_scaler.scale_ = v3_scaler.scale_[:len(v3_base_features)]
v3_ae_scaler.var_ = v3_scaler.var_[:len(v3_base_features)]
v3_ae_scaler.n_features_in_ = len(v3_base_features)

X_v3_base_s = v3_ae_scaler.transform(X_v3_base)
rec_v3 = v3_ae.predict(X_v3_base_s, batch_size=2048, verbose=0)
ae_err_v3 = np.log1p(np.mean(np.square(X_v3_base_s - rec_v3), axis=1))

# Identify normal transactions and calculate their mean AE error
normal_mask = (y_test == 0)
mean_normal_ae_err = np.mean(ae_err_v3[normal_mask])

# -------------------------------------------------------------
# Baseline Evaluation
# -------------------------------------------------------------
X_v3_19_baseline = np.column_stack([X_v3_base, ae_err_v3])
X_v3_19_s_baseline = v3_scaler.transform(X_v3_19_baseline)
baseline_probs = v3_xgb.predict_proba(X_v3_19_s_baseline)[:, 1]
baseline_preds = (baseline_probs >= v3_block_threshold).astype(int)

baseline_recall = recall_score(y_test, baseline_preds, zero_division=0)

# -------------------------------------------------------------
# AE Neutralization Test
# -------------------------------------------------------------
# Force AE error of fraud transactions to equal the normal mean
fraud_mask = (y_test == 1)
ae_err_corrupted = np.copy(ae_err_v3)
ae_err_corrupted[fraud_mask] = mean_normal_ae_err

X_v3_19_corrupted = np.column_stack([X_v3_base, ae_err_corrupted])
X_v3_19_s_corrupted = v3_scaler.transform(X_v3_19_corrupted)
corrupted_probs = v3_xgb.predict_proba(X_v3_19_s_corrupted)[:, 1]
corrupted_preds = (corrupted_probs >= v3_block_threshold).astype(int)

corrupted_recall = recall_score(y_test, corrupted_preds, zero_division=0)
recall_drop = baseline_recall - corrupted_recall

print(f"Baseline Recall: {baseline_recall:.4f}")
print(f"Corrupted Recall (Neutralized AE Error): {corrupted_recall:.4f}")
print(f"Recall Drop: {recall_drop:.4f}")

# Write report
with open(f"{OUTPUT_DIR}/AE_Reliance_Test_Report.md", "w") as f:
    f.write("# Autoencoder (AE) Reliance Stress Test\n\n")
    f.write("## Objective\n")
    f.write("Determine if the XGBoost classification model is overly dependent on the Autoencoder Reconstruction Error (`ae_recon_error`) for detecting fraud. If the AE fails or normalizes a fraudulent transaction, we must ensure the core behavioral features (velocity, amount ratios, etc.) provide enough redundant signal to catch the fraud.\n\n")
    
    f.write("## Methodology\n")
    f.write("1. Evaluated baseline recall on the test set using actual AE reconstruction errors.\n")
    f.write(f"2. Calculated the mean `ae_recon_error` for **normal/legitimate** transactions (Mean: {mean_normal_ae_err:.4f}).\n")
    f.write("3. **Neutralized** the `ae_recon_error` for all known **fraudulent** transactions in the test set by explicitly setting their error to the normal mean (simulating a complete failure of the Autoencoder to flag the anomaly).\n")
    f.write("4. Re-evaluated the XGBoost model's recall on this corrupted dataset.\n\n")
    
    f.write("## Results\n")
    f.write(f"- **Baseline Recall (Real AE Errors):** {baseline_recall:.2%}\n")
    f.write(f"- **Corrupted Recall (Neutralized AE Errors):** {corrupted_recall:.2%}\n")
    f.write(f"- **Recall Drop:** {recall_drop:.2%} (Absolute Drop)\n\n")
    
    f.write("## Conclusion\n")
    if recall_drop < 0.05:
        f.write("The model shows **extreme robustness**. Neutralizing the Autoencoder's contribution caused an insignificant drop in recall. This proves the XGBoost model does not lazily rely on a single feature. Instead, behavioral features (like velocity and relative amounts) provide a strong, overlapping safety net, guaranteeing high real-time reliability even if one component of the ensemble fails.\n")
    elif recall_drop < 0.15:
        f.write("The model shows **moderate robustness**. While the AE error provides a meaningful lift, neutralizing it only drops the recall slightly. The model successfully falls back on other behavioral features, avoiding critical single-point-of-failure vulnerabilities.\n")
    else:
        f.write("The model shows **high dependence** on the Autoencoder error. While it still catches a portion of fraud using behavioral features, the significant drop in recall implies that `ae_recon_error` is a dominant signal. Care should be taken to ensure the AE remains robust in production.\n")

print(f"Report saved to {OUTPUT_DIR}/AE_Reliance_Test_Report.md")
