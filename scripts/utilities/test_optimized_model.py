"""
Test the optimized fraud detection model on demo files
to verify improved recall and precision.
"""
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.final_ensemble_inference import ensemble_predict

print("="*80)
print(" OPTIMIZED MODEL TEST - DEMO FILES VALIDATION")
print("="*80)

# Test File 1: paysim_demo_final.csv
print("\n[TEST 1] paysim_demo_final.csv")
print("-"*80)

df1 = pd.read_csv('data_generation/paysim_demo_final.csv')
actual_frauds_1 = df1['has_balance_mismatch'].sum()

detected_1 = 0
missed_1 = []
false_positives_1 = 0

for idx, row in df1.iterrows():
    result = ensemble_predict(
        transaction_type="paysim",
        raw_df=pd.DataFrame([row]),
        lstm_sequence=None
    )
    
    is_fraud = row['has_balance_mismatch'] == 1
    detected_fraud = result['decision']
    
    if detected_fraud and is_fraud:
        detected_1 += 1
    elif detected_fraud and not is_fraud:
        false_positives_1 += 1
    elif not detected_fraud and is_fraud:
        missed_1.append((idx, row['amount'], result['score']))

print(f"Actual Frauds: {actual_frauds_1}")
print(f"Detected: {detected_1}")
print(f"Missed: {len(missed_1)}")
print(f"False Positives: {false_positives_1}")

if len(missed_1) > 0:
    print("\n⚠️ MISSED FRAUDS:")
    for idx, amount, score in missed_1:
        print(f"  Row {idx}: ${amount:,.2f} (score: {score:.4f})")
else:
    print("\n✅ ALL FRAUDS DETECTED!")

recall_1 = (detected_1 / actual_frauds_1 * 100) if actual_frauds_1 > 0 else 0
print(f"\nRecall: {recall_1:.1f}%")

# Test File 2: paysim_demo.csv
print("\n" + "="*80)
print("[TEST 2] paysim_demo.csv")
print("-"*80)

df2 = pd.read_csv('data_generation/paysim_demo.csv')
actual_frauds_2 = df2['has_balance_mismatch'].sum()

detected_2 = 0
missed_2 = []
false_positives_2 = 0

for idx, row in df2.iterrows():
    result = ensemble_predict(
        transaction_type="paysim",
        raw_df=pd.DataFrame([row]),
        lstm_sequence=None
    )
    
    is_fraud = row['has_balance_mismatch'] == 1
    detected_fraud = result['decision']
    
    if detected_fraud and is_fraud:
        detected_2 += 1
    elif detected_fraud and not is_fraud:
        false_positives_2 += 1
    elif not detected_fraud and is_fraud:
        missed_2.append((idx, row['amount'], result['score']))

print(f"Actual Frauds: {actual_frauds_2}")
print(f"Detected: {detected_2}")
print(f"Missed: {len(missed_2)}")
print(f"False Positives: {false_positives_2}")

if len(missed_2) > 0:
    print("\n⚠️ MISSED FRAUDS:")
    for idx, amount, score in missed_2:
        print(f"  Row {idx}: ${amount:,.2f} (score: {score:.4f})")
else:
    print("\n✅ ALL FRAUDS DETECTED!")

recall_2 = (detected_2 / actual_frauds_2 * 100) if actual_frauds_2 > 0 else 0
print(f"\nRecall: {recall_2:.1f}%")

# Combined Results
print("\n" + "="*80)
print(" COMBINED RESULTS")
print("="*80)

total_frauds = actual_frauds_1 + actual_frauds_2
total_detected = detected_1 + detected_2
total_missed = len(missed_1) + len(missed_2)
total_fp = false_positives_1 + false_positives_2

print(f"Total Frauds: {total_frauds}")
print(f"Total Detected: {total_detected}")
print(f"Total Missed: {total_missed}")
print(f"Total False Positives: {total_fp}")

combined_recall = (total_detected / total_frauds * 100) if total_frauds > 0 else 0
combined_precision = (total_detected / (total_detected + total_fp) * 100) if (total_detected + total_fp) > 0 else 0

print(f"\n📊 Performance Metrics:")
print(f"   Recall: {combined_recall:.1f}%")
print(f"   Precision: {combined_precision:.1f}%")

if combined_recall == 100 and combined_precision >= 90:
    print("\n🎉 SUCCESS! Model achieves target performance!")
elif combined_recall >= 95:
    print("\n✅ GOOD! High recall achieved, precision acceptable.")
else:
    print("\n⚠️ WARNING: Performance below target.")

print("="*80)
