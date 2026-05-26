import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# ===============================
# PATHS
# ===============================
DATA_PATH = "Fraud_Detection_Model_Paysim_CC/data/creditcard_test.csv"

AE_MODEL_PATH = "Fraud_Detection_Model_Paysim_CC/models/cc_ae_model.keras"
AE_SCALER_PATH = "Fraud_Detection_Model_Paysim_CC/models/cc_ae_scaler.pkl"
AE_FEATURES_PATH = "Fraud_Detection_Model_Paysim_CC/models/cc_ae_features.pkl"

XGB_AE_MODEL_PATH = "Fraud_Detection_Model_Paysim_CC/models/cc_ae_xgb_model.pkl"
SCALER_PATH = "Fraud_Detection_Model_Paysim_CC/models/cc_scaler.pkl"
FEATURES_PATH = "Fraud_Detection_Model_Paysim_CC/models/cc_features.pkl"
THRESHOLD = 0.5

print("🔥 Evaluating Credit Card XGBoost + Autoencoder (FINAL MODEL)")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH)
y = df["isFraud"].values

# Case-insensitive map
df_cols = {c.lower(): c for c in df.columns}

# ===============================
# AUTOENCODER FEATURE
# ===============================
ae = tf.keras.models.load_model(AE_MODEL_PATH, compile=False)
ae_scaler = joblib.load(AE_SCALER_PATH)
ae_features = joblib.load(AE_FEATURES_PATH)

ae_cols = [df_cols[f] for f in ae_features]
X_ae = ae_scaler.transform(df[ae_cols].values)

recon = ae.predict(X_ae, batch_size=1024, verbose=0)
ae_error = np.mean(np.square(X_ae - recon), axis=1)

# ===============================
# BASE FEATURES (SCALED ONLY)
# ===============================
base_features = joblib.load(FEATURES_PATH)   # 32 features
scaler = joblib.load(SCALER_PATH)

X_base = np.column_stack([df[df_cols[f]].values for f in base_features])
X_base_scaled = scaler.transform(X_base)

# ===============================
# FINAL FEATURE MATRIX (33)
# ===============================
X_sup = np.column_stack([X_base_scaled, ae_error])

print("Final feature shape:", X_sup.shape)
print("Model expects:", joblib.load(XGB_AE_MODEL_PATH).get_booster().num_features())

# ===============================
# PREDICT
# ===============================
xgb = joblib.load(XGB_AE_MODEL_PATH)
probs = xgb.predict_proba(X_sup)[:, 1]
y_pred = (probs >= THRESHOLD).astype(int)

# ===============================
# METRICS
# ===============================
print("\nCREDIT CARD XGB + AE CONFUSION MATRIX")
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))
