import joblib
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("Fraud_Detection_Model_Paysim_CC/data/paysim_augmented_with_frauds.csv")
BASELINE_MODELS = Path("Fraud_Detection_Model_Paysim_CC/models")
STAGE2_MODELS   = Path("Fraud_Detection_Model_Paysim_CC/models/stage2_experiment")


print("\n=== Stress Test: Synthetic Frauds ===")

df = pd.read_csv(DATA)

# baseline
base_xgb = joblib.load(BASELINE_MODELS / "paysim_xgb.joblib")
base_pre = joblib.load(BASELINE_MODELS / "paysim_preproc.joblib")
base_feat = joblib.load(BASELINE_MODELS / "paysim_stage2_features.pkl")

# stage2
s2_xgb = joblib.load(STAGE2_MODELS / "paysim_xgb_stage2.pkl")
s2_scaler = joblib.load(STAGE2_MODELS / "paysim_stage2_scaler.pkl")
s2_feat = joblib.load(STAGE2_MODELS / "paysim_stage2_features.pkl")
s2_thr = float(np.load(STAGE2_MODELS / "paysim_stage2_threshold.npy"))


Xs = s2_scaler.transform(df[s2_feat])
ps = s2_xgb.predict_proba(Xs)[:, 1]
rs = ((ps >= s2_thr) & (df["isFraud"] == 1)).mean()

print(f"Baseline recall on synthetic frauds : {rb:.3f}")
print(f"Stage-2 recall on synthetic frauds  : {rs:.3f}")
