import pandas as pd
import numpy as np
from src.final_ensemble_inference import ensemble_predict

print("="*80)
print(" CREDIT CARD ENSEMBLE - PERFORMANCE VALIDATION")
print("="*80)

# Load demo CSV with ground truth
demo_df = pd.read_csv('data_generation/creditcard_demo_final.csv')

ground_truth = []
predictions = []
scores = []
explanations = []

print(f"\n🧪 Testing ENSEMBLE on {len(demo_df)} transactions...")
print(f"   Ground Truth Frauds: {demo_df['is_fraud'].sum()}")

# Test each transaction
for idx, row in demo_df.iterrows():
    # Store ground truth
    ground_truth.append(int(row['is_fraud']))
    
    # Create test dataframe (without is_fraud column)
    test_row = row.drop('is_fraud')
    test_df = pd.DataFrame([test_row])
    
    # Get prediction using ENSEMBLE
    result = ensemble_predict('creditcard', test_df)
    predictions.append(1 if result['decision'] else 0)
    scores.append(result['score'])
    explanations.append(result['explanation'])

# Verify ensemble is being used
sample_explanation = explanations[0]
if 'Ensemble' in sample_explanation:
    print(f"✅ Ensemble is ACTIVE!")
    print(f"   Sample: {sample_explanation}")
else:
    print(f"❌ ERROR: Ensemble not active, still using old prediction")
    print(f"   Sample: {sample_explanation}")

# Calculate metrics
tp = sum([1 for i in range(len(predictions)) if ground_truth[i] == 1 and predictions[i] == 1])
fp = sum([1 for i in range(len(predictions)) if ground_truth[i] == 0 and predictions[i] == 1])
tn = sum([1 for i in range(len(predictions)) if ground_truth[i] == 0 and predictions[i] == 0])
fn = sum([1 for i in range(len(predictions)) if ground_truth[i] == 1 and predictions[i] == 0])

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n📊 ENSEMBLE RESULTS:")
print(f"   Detected Frauds: {sum(predictions)}")
print(f"   Detected Legit:  {len(predictions) - sum(predictions)}")

print(f"\n📈 Confusion Matrix:")
print(f"   True Positives (TP):  {tp}")
print(f"   False Positives (FP): {fp}")
print(f"   True Negatives (TN):  {tn}")
print(f"   False Negatives (FN): {fn}")

print(f"\n🎯 Performance Metrics:")
print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"   F1-Score:  {f1:.4f}")

# Compare to baseline (previous: 15.4% precision, 80% recall)
print(f"\n📊 Comparison to Baseline (Single XGBoost, threshold=0.0235):")
print(f"   Precision: {precision*100:.1f}% vs 15.4% → {(precision/0.154-1)*100:+.0f}% change")
print(f"   Recall:    {recall*100:.1f}% vs 80.0% → {(recall/0.80-1)*100:+.0f}% change")
print(f"   F1-Score:  {f1:.2f} vs 0.26 → {(f1/0.26-1)*100:+.0f}% change")

# Show missed frauds if any
if fn > 0:
    print(f"\n❌ Missed Frauds ({fn}):")
    missed_indices = [i for i in range(len(predictions)) if ground_truth[i] == 1 and predictions[i] == 0]
    for idx in missed_indices:
        print(f"   Row {idx}: Score = {scores[idx]:.4f}")
        print(f"             {explanations[idx]}")

if fp > 0:
    print(f"\n⚠️  False Positives ({fp}):")
    fp_indices = [i for i in range(len(predictions)) if ground_truth[i] == 0 and predictions[i] == 1]
    for idx in fp_indices[:5]:  # Show first 5
        print(f"   Row {idx}: Score = {scores[idx]:.4f}")

print("\n" + "="*80)
