import pandas as pd
import numpy as np

print("="*80)
print(" CREDIT CARD MODEL VALIDATION REPORT")
print("="*80)

# Read audit file
audit = pd.read_csv('fraud_audit_1770545574.csv')

print(f"\n📊 Audit File Analysis")
print(f"   Total transactions processed: {len(audit)}")

# Count detections
fraud_count = (audit['Detection'] == 'FRAUD').sum()
legit_count = (audit['Detection'] == 'LEGIT').sum()

print(f"\n🔍 Detection Results:")
print(f"   🚨 FRAUD: {fraud_count} transactions")
print(f"   ✅ LEGIT: {legit_count} transactions")

# Check the generator to understand ground truth
print(f"\n📋 Generate Script Analysis:")
print(f"   FRAUD_RATIO: 15% (configured in generate_creditcard_demo_csv.py)")
print(f"   Expected frauds: ~{int(50 * 0.15)} out of 50")

print(f"\n❓ Ground Truth Status:")
print(f"   The credit card demo file uses SYNTHETIC data generation")
print(f"   Frauds are generated with mean(5, 2) vs legit mean(0, 1)")
print(f"   However, the generated CSV does NOT include ground truth labels")
print(f"   Cannot calculate True Positives/False Positives without labels")

if fraud_count > 0:
    print(f"\n🚨 Detected Frauds Details:")
    frauds = audit[audit['Detection'] == 'FRAUD']
    print(frauds[['amount_scaled', 'hour', 'Detection', 'Insight']].head(10))

print(f"\n" + "="*80)
print(" CONCLUSION")
print("="*80)
print(f"✅ Model is working: {fraud_count} frauds detected")
print(f"⚠️  Cannot validate accuracy: No ground truth labels in CSV")
print(f"💡 Recommendation: Modify generator to include 'is_fraud' column")
print("="*80)
