import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report

DATA_PATH = "Fraud_Detection_Model_Paysim_CC/data/paysim_test.csv"

MODEL_PATH = "Fraud_Detection_Model_Paysim_CC/artifacts/paysim_rf.joblib"
PREPROC_PATH = "Fraud_Detection_Model_Paysim_CC/artifacts/paysim_preproc.joblib"

print("🌲 Evaluating PaySim Random Forest (Baseline)")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(DATA_PATH)

y = df["isFraud"].values

# ===============================
# LOAD PREPROCESSOR
# ===============================
preproc = joblib.load(PREPROC_PATH)

# 🔥 CRITICAL FIX: align feature columns
expected_features = preproc.feature_names_in_
X_raw = df[expected_features]

print("Using features:", list(expected_features))
print("Total features:", len(expected_features))

# ===============================
# TRANSFORM
# ===============================
X = preproc.transform(X_raw)

# ===============================
# LOAD MODEL
# ===============================
rf = joblib.load(MODEL_PATH)

print("Model expects:", rf.n_features_in_)
print("Input features:", X.shape[1])

# ===============================
# PREDICT
# ===============================
y_pred = rf.predict(X)

# ===============================
# METRICS
# ===============================
print("\nPAYSim RANDOM FOREST CONFUSION MATRIX")
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))
