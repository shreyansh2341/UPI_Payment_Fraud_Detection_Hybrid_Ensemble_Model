# Calculate metrics and optimize threshold
import pandas as pd
import numpy as np
import joblib

# Load export data
export = pd.read_csv('2026-02-08T18-41_export.csv')

# Metrics from current results
tp = len(export[(export['is_fraud']==1) & (export['Detection']=='FRAUD')])
fp = len(export[(export['is_fraud']==0) & (export['Detection']=='FRAUD')])
fn = len(export[(export['is_fraud']==1) & (export['Detection']=='LEGIT')])
tn = len(export[(export['is_fraud']==0) & (export['Detection']=='LEGIT')])

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("="*60)
print("CURRENT RESULTS (threshold=0.0097)")
print("="*60)
print(f"True Positives (fraud detected):   {tp}")
print(f"False Positives (legit as fraud):  {fp}")
print(f"True Negatives (legit as legit):  {tn}")
print(f"False Negatives (fraud missed):    {fn}")
print(f"\nPrecision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-Score:  {f1:.4f}")

# Now find optimal threshold
print("\n" + "="*60)
print("THRESHOLD OPTIMIZATION")
print("="*60)

# Parse scores from Insight column
def extract_score(insight):
    try:
        return float(insight.split('=')[1].split(' ')[0])
    except:
        return 0

export['score'] = export['Insight'].apply(extract_score)

# Get fraud and legit scores
fraud_scores = export[export['is_fraud']==1]['score'].values
legit_scores = export[export['is_fraud']==0]['score'].values

print(f"\nFraud scores: {[f'{s:.4f}' for s in sorted(fraud_scores)]}")
print(f"Min fraud score: {min(fraud_scores):.6f}")
print(f"Max legit score: {max(legit_scores):.6f}")
print(f"Legit 95th pct:  {np.percentile(legit_scores, 95):.6f}")

# Find best threshold
for t in np.arange(0.005, 0.015, 0.001):
    tp = sum(s >= t for s in fraud_scores)
    fp = sum(s >= t for s in legit_scores)
    fn = sum(s < t for s in fraud_scores)
    tn = sum(s < t for s in legit_scores)
    p = tp/(tp+fp) if (tp+fp)>0 else 0
    r = tp/(tp+fn) if (tp+fn)>0 else 0
    f = 2*p*r/(p+r) if (p+r)>0 else 0
    print(f"t={t:.4f}: TP={tp}, FP={fp}, P={p:.2%}, R={r:.2%}, F1={f:.4f}")

# The minimum fraud score appears to be the key threshold
best_threshold = min(fraud_scores) - 0.0001
print(f"\nOptimal threshold: {best_threshold:.6f}")
print("(just below minimum fraud score to ensure 100% recall)")

# Save new threshold
np.save('models/cc_ensemble_threshold.npy', np.array([best_threshold]))
print(f"\nSaved new threshold: {best_threshold:.6f}")
