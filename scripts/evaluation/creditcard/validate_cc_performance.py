import pandas as pd
import numpy as np

print("="*80)
print(" CREDIT CARD MODEL - PERFORMANCE VALIDATION")
print("="*80)

# Load demo CSV with ground truth
demo_df = pd.read_csv('data_generation/creditcard_demo_final.csv')
print(f"\n📊 Demo CSV Loaded")
print(f"   Total rows: {len(demo_df)}")
print(f"   Columns: {demo_df.columns.tolist()[:5]}... + is_fraud")

# Ground truth
actual_frauds = demo_df['is_fraud'].sum()
actual_legit = len(demo_df) - actual_frauds
print(f"\n🎯 Ground Truth:")
print(f"   Actual FRAUD: {actual_frauds} ({actual_frauds/len(demo_df)*100:.1f}%)")
print(f"   Actual LEGIT: {actual_legit} ({actual_legit/len(demo_df)*100:.1f}%)")

# Load audit report (if exists)
try:
    audit_df = pd.read_csv('fraud_audit_1770545574.csv')
    
    # Merge with ground truth
    demo_df_with_detection = demo_df.copy()
    demo_df_with_detection['detected'] = (audit_df['Detection'] == 'FRAUD').astype(int)
    
    # Calculate metrics
    tp = ((demo_df_with_detection['is_fraud'] == 1) & (demo_df_with_detection['detected'] == 1)).sum()
    fp = ((demo_df_with_detection['is_fraud'] == 0) & (demo_df_with_detection['detected'] == 1)).sum()
    tn = ((demo_df_with_detection['is_fraud'] == 0) & (demo_df_with_detection['detected'] == 0)).sum()
    fn = ((demo_df_with_detection['is_fraud'] == 1) & (demo_df_with_detection['detected'] == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    detected_frauds = (audit_df['Detection'] == 'FRAUD').sum()
    
    print(f"\n🔍 Model Detections (Old Audit):")
    print(f"   Detected FRAUD: {detected_frauds}")
    print(f"   Detected LEGIT: {len(audit_df) - detected_frauds}")
    
    print(f"\n📈 Performance Metrics:")
    print(f"   True Positives (TP):  {tp}")
    print(f"   False Positives (FP): {fp}")
    print(f"   True Negatives (TN):  {tn}")
    print(f"   False Negatives (FN): {fn}")
    print(f"\n   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1-Score:  {f1:.4f}")
    
    print(f"\n⚠️  Note: This audit is from the OLD CSV without ground truth labels")
    print(f"    Please upload the NEW creditcard_demo_final.csv to get accurate results")
    
except FileNotFoundError:
    print(f"\n⚠️  No audit file found yet")
    print(f"   Please upload creditcard_demo_final.csv via frontend to generate audit")

print("\n" + "="*80)
print(" NEXT STEPS")
print("="*80)
print("1. Upload the NEW creditcard_demo_final.csv via frontend")
print("2. Check the new audit report")
print("3. Re-run this script to see TRUE performance metrics")
print("="*80)
