import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# ===============================
# PATHS
# ===============================
DATA_PATH = "data/creditcard_test.csv"

MODEL_PATH = "models/cc_xgb_model.pkl"
SCALER_PATH = "models/cc_scaler.pkl"
FEATURES_PATH = "models/cc_features.pkl"
THRESH_PATH = "models/cc_threshold.npy"

print("Evaluating Credit Card XGBoost")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH)
y = df["isFraud"].values

# ===============================
# LOAD MODEL ARTIFACTS
# ===============================
features = joblib.load(FEATURES_PATH)      # expects lowercase v1..v28
scaler = joblib.load(SCALER_PATH)
threshold = np.load(THRESH_PATH)[0]

# ===============================
# 🔥 FIX: ALIGN COLUMN NAMES
# ===============================
df_cols = {c.lower(): c for c in df.columns}

mapped_features = [df_cols[f] for f in features if f in df_cols]

missing = set(features) - set(df_cols.keys())
if missing:
    raise ValueError(f"Missing required features: {missing}")

X = scaler.transform(df[mapped_features].values)

# ===============================
# LOAD MODEL
# ===============================
xgb = joblib.load(MODEL_PATH)

probs = xgb.predict_proba(X)[:, 1]
y_pred = (probs >= threshold).astype(int)

# ===============================
# METRICS
# ===============================
print("\nCREDIT CARD XGBOOST CONFUSION MATRIX")
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))
