"""
Analyze ensemble scores on demo data and set reasonable threshold
"""
import pandas as pd
import numpy as np
import joblib

print("="*70)
print("ANALYZING ENSEMBLE SCORES ON DEMO DATA")
print("="*70)

# Load demo data
demo = pd.read_csv('data_generation/creditcard_demo_final.csv')
print(f"\nDemo dataset: {len(demo)} rows, {demo['is_fraud'].sum()} actual frauds")

# Load models
xgb = joblib.load('models/cc_xgb_model.pkl')
rf = joblib.load('models/cc_rf_model.pkl')
scaler = joblib.load('models/cc_scaler.pkl')
features = joblib.load('models/cc_features.pkl')
weights = np.load('models/cc_ensemble_weights.npy')
current_threshold = np.load('models/cc_ensemble_threshold.npy')[0]

print(f"\nCurrent Threshold: {current_threshold:.6f}")
print(f"Weights: XGB={weights[0]}, RF={weights[1]}")

# Get predictions for all rows
results = []
for idx in range(len(demo)):
    row = demo.iloc[idx]
    is_fraud = row['is_fraud']
    
    X = pd.DataFrame([row.drop('is_fraud')])
    X.columns = X.columns.str.lower()
    X = X[features]
    X_scaled = scaler.transform(X)
    
    prob_xgb = xgb.predict_proba(X_scaled)[0, 1]
    prob_rf = rf.predict_proba(X_scaled)[0, 1]
    prob_ensemble = weights[0]*prob_xgb + weights[1]*prob_rf
    
    results.append({
        'idx': idx,
        'is_fraud': is_fraud,
        'prob_xgb': prob_xgb,
        'prob_rf': prob_rf,
        'prob_ensemble': prob_ensemble
    })

results_df = pd.DataFrame(results)

# Show fraud vs legit statistics
print("\n" + "-"*70)
print("PROBABILITY STATISTICS")
print("-"*70)

frauds = results_df[results_df['is_fraud'] == 1]
legits = results_df[results_df['is_fraud'] == 0]

print(f"\nFRAUD transactions ({len(frauds)}):")
print(f"  Ensemble prob range: {frauds['prob_ensemble'].min():.4f} - {frauds['prob_ensemble'].max():.4f}")
print(f"  Ensemble prob mean: {frauds['prob_ensemble'].mean():.4f}")
print(f"  XGB prob mean: {frauds['prob_xgb'].mean():.4f}")
print(f"  RF prob mean: {frauds['prob_rf'].mean():.4f}")

print(f"\nLEGIT transactions ({len(legits)}):")
print(f"  Ensemble prob range: {legits['prob_ensemble'].min():.4f} - {legits['prob_ensemble'].max():.4f}")
print(f"  Ensemble prob mean: {legits['prob_ensemble'].mean():.4f}")

# Show all fraud transactions
print("\n" + "-"*70)
print("ALL FRAUD TRANSACTIONS")
print("-"*70)
for _, row in frauds.iterrows():
    pred = "DETECTED" if row['prob_ensemble'] >= current_threshold else "MISSED"
    print(f"Row {int(row['idx'])}: Ensemble={row['prob_ensemble']:.4f} (XGB={row['prob_xgb']:.4f}, RF={row['prob_rf']:.4f}) -> {pred}")

# Suggest threshold
max_legit = legits['prob_ensemble'].max()
min_fraud = frauds['prob_ensemble'].min()

print("\n" + "-"*70)
print("THRESHOLD ANALYSIS")
print("-"*70)
print(f"Highest legit score: {max_legit:.4f}")
print(f"Lowest fraud score:  {min_fraud:.4f}")

# Find ideal threshold
if min_fraud > max_legit:
    ideal_threshold = (max_legit + min_fraud) / 2
    print(f"\nGood separation! Ideal threshold: {ideal_threshold:.4f}")
else:
    print(f"\nOverlap detected! Need to choose trade-off")
    # Use percentile-based approach
    ideal_threshold = np.percentile(results_df['prob_ensemble'], 90)  # Top 10% as fraud
    print(f"Suggested threshold (90th percentile): {ideal_threshold:.4f}")

# Save new threshold
print("\n" + "-"*70)
print("SETTING NEW THRESHOLD")
print("-"*70)

# Use threshold that catches all frauds if possible
new_threshold = min(min_fraud * 0.9, 0.5)  # 90% of lowest fraud score, cap at 0.5
print(f"New threshold: {new_threshold:.6f}")

np.save('models/cc_ensemble_threshold.npy', np.array([new_threshold]))
print("Saved to models/cc_ensemble_threshold.npy")

# Test with new threshold
print("\n" + "-"*70)
print("PREDICTIONS WITH NEW THRESHOLD")
print("-"*70)

tp = len(frauds[frauds['prob_ensemble'] >= new_threshold])
fp = len(legits[legits['prob_ensemble'] >= new_threshold])
fn = len(frauds) - tp
tn = len(legits) - fp

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"True Positives:  {tp}")
print(f"False Positives: {fp}")
print(f"True Negatives:  {tn}")
print(f"False Negatives: {fn}")
print(f"\nPrecision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-Score:  {f1:.4f}")

print("\n" + "="*70)
print("RESTART BACKEND TO USE NEW THRESHOLD!")
print("="*70)
