"""
Analyze actual credit card ensemble results from audit file
"""
import pandas as pd

# Read the latest audit file
audit_df = pd.read_csv('fraud_audit_1770545574.csv')

print("="*80)
print(" CREDIT CARD ENSEMBLE - ACTUAL RESULTS ANALYSIS")
print("="*80)

# Load ground truth
demo_df = pd.read_csv('data_generation/creditcard_demo_final.csv')
ground_truth_frauds = demo_df['is_fraud'].values

# Check if audit matches demo
if len(audit_df) != len(demo_df):
    print(f"\n⚠️  WARNING: Audit has {len(audit_df)} rows, demo has {len(demo_df)} rows")
else:
    print(f"\n✅ Audit and demo have same number of rows: {len(audit_df)}")

# Analyze detections
detections = (audit_df['Detection'] == 'FRAUD').values
detected_count = detections.sum()

print(f"\n📊 Detection Summary:")
print(f"   Total Transactions: {len(audit_df)}")
print(f"   Detected as FRAUD: {detected_count}")
print(f"   Detected as LEGIT: {len(audit_df) - detected_count}")

# Calculate metrics with ground truth
if len(audit_df) == len(demo_df):
    tp = sum([1 for i in range(len(detections)) if ground_truth_frauds[i] == 1 and detections[i] == 1])
    fp = sum([1 for i in range(len(detections)) if ground_truth_frauds[i] == 0 and detections[i] == 1])
    tn = sum([1 for i in range(len(detections)) if ground_truth_frauds[i] == 0 and detections[i] == 0])
    fn = sum([1 for i in range(len(detections)) if ground_truth_frauds[i] == 1 and detections[i] == 0])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n🎯 Ground Truth Comparison:")
    print(f"   Actual Frauds: {ground_truth_frauds.sum()}")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"   True Positives (TP):  {tp}")
    print(f"   False Positives (FP): {fp}")
    print(f"   True Negatives (TN):  {tn}")
    print(f"   False Negatives (FN): {fn}")
    
    print(f"\n⭐ Performance Metrics:")
    print(f"   Precision: {precision:.4f} ({precision*100:.1f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.1f}%)")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Analyze XGB Override usage
    xgb_override_count = audit_df['Insight'].str.contains('XGB Override', na=False).sum()
    print(f"\n🔍 XGB Override Analysis:")
    print(f"   Detections using XGB Override: {xgb_override_count} / {detected_count}")
    print(f"   Regular ensemble detections: {detected_count - xgb_override_count} / {detected_count}")
    
    if xgb_override_count > 0:
        print(f"\n⚠️  WARNING: XGB Override should NOT be used for credit card!")
        print(f"   This indicates PaySim logic is contaminating credit card predictions")

print("\n" + "="*80)
