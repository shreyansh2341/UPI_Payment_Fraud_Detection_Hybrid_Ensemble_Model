import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# ===============================
# PATHS
# ===============================
DATA_PATH = "Fraud_Detection_Model_Paysim_CC/data/paysim_test.csv"

# AE artifacts
AE_MODEL_PATH = "Fraud_Detection_Model_Paysim_CC/artifacts/paysim_ae_model.keras"
AE_PREPROC_PATH = "Fraud_Detection_Model_Paysim_CC/artifacts/paysim_ae_preproc.joblib"
AE_THRESH_PATH = "Fraud_Detection_Model_Paysim_CC/artifacts/paysim_ae_thresh.npy"

# Supervised model artifacts
SUP_PREPROC_PATH = "Fraud_Detection_Model_Paysim_CC/artifacts/paysim_sup_preproc.joblib"
XGB_AE_MODEL_PATH = "Fraud_Detection_Model_Paysim_CC/artifacts/paysim_xgb_ae.joblib"
THRESHOLD = 0.5

print("🔥 Evaluating PaySim XGBoost + Autoencoder (FINAL MODEL)")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH)
y = df["isFraud"].values
X_raw = df.drop(columns=["isFraud"])

# ===============================
# LOAD AUTOENCODER
# ===============================
ae = tf.keras.models.load_model(AE_MODEL_PATH, compile=False)
ae_preproc = joblib.load(AE_PREPROC_PATH)
ae_thresh = np.load(AE_THRESH_PATH)[0]

# ===============================
# COMPUTE AE FEATURES
# ===============================
X_ae = ae_preproc.transform(X_raw)

recon = ae.predict(X_ae, batch_size=1024, verbose=0)
ae_error = np.mean(np.square(X_ae - recon), axis=1)
ae_anomaly = (ae_error >= ae_thresh).astype(int)

df["ae_error"] = ae_error
df["ae_anomaly"] = ae_anomaly

print("AE features added:")
print(df[["ae_error", "ae_anomaly"]].describe())

# ===============================
# SUPERVISED PREPROCESSING
# ===============================
sup_preproc = joblib.load(SUP_PREPROC_PATH)
expected_features = sup_preproc.feature_names_in_
X_sup_raw = df[expected_features]

X_sup = sup_preproc.transform(X_sup_raw)

# ===============================
# LOAD XGB + AE MODEL
# ===============================
xgb = joblib.load(XGB_AE_MODEL_PATH)

probs = xgb.predict_proba(X_sup)[:, 1]
y_pred = (probs >= THRESHOLD).astype(int)

# ===============================
# METRICS
# ===============================
print("\nPAYSim XGB + AE CONFUSION MATRIX")
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))
