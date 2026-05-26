import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

DATA_PATH = "data/paysim_test.csv"

MODEL_PATH = "models/paysim_xgb.joblib"
PREPROC_PATH = "models/paysim_preproc.joblib"

THRESHOLD = 0.5

print("Evaluating PaySim XGBoost (Baseline)")

df = pd.read_csv(DATA_PATH)

y = df["isFraud"].values

preproc = joblib.load(PREPROC_PATH)
expected_features = preproc.feature_names_in_
X_raw = df[expected_features]

X = preproc.transform(X_raw)

xgb = joblib.load(MODEL_PATH)

probs = xgb.predict_proba(X)[:, 1]
y_pred = (probs >= THRESHOLD).astype(int)

print("\nPAYSim XGBOOST CONFUSION MATRIX")
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))
