import joblib, json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score

DATA = Path("Fraud_Detection_Model_Paysim_CC/data/paysim_test.csv")
MODELS = Path("Fraud_Detection_Model_Paysim_CC/models")
OUT = Path("Fraud_Detection_Model_Paysim_CC/results/baseline_metrics.json")

xgb = joblib.load(MODELS / "paysim_xgb.joblib")
pre = joblib.load(MODELS / "paysim_preproc.joblib")
features = joblib.load(MODELS / "paysim_stage2_features.pkl")

df = pd.read_csv(DATA)
X = pre.transform(df[features])
y = df["isFraud"].values

probs = xgb.predict_proba(X)[:, 1]
preds = (probs >= 0.5).astype(int)

results = {
    "roc_auc": roc_auc_score(y, probs),
    "report": classification_report(y, preds, output_dict=True),
}

OUT.parent.mkdir(exist_ok=True)
json.dump(results, open(OUT, "w"), indent=2)

print("✅ Baseline evaluation complete")
print(json.dumps(results, indent=2))
