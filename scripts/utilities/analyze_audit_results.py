import pandas as pd

# Load audit files
audit1 = pd.read_csv('fraud_audit_1770543553.csv')
audit2 = pd.read_csv('fraud_audit_1770543583.csv')

print("="*80)
print(" FRAUD DETECTION MODEL VALIDATION REPORT")
print("="*80)

# Analyze each audit file
for i, (audit, file_name) in enumerate([(audit1, 'fraud_audit_1770543553.csv'), 
                                          (audit2, 'fraud_audit_1770543583.csv')], 1):
    print(f"\n{'='*80}")
    print(f" AUDIT FILE {i}: {file_name}")
    print(f"{'='*80}")
    
    # Basic stats
    total = len(audit)
    fraud_detected = (audit['Detection'] == 'FRAUD').sum()
    legit_detected = (audit['Detection'] == 'LEGIT').sum()
    errors = (audit['Detection'] == 'ERROR').sum()
    
    print(f"\n📊 Detection Summary:")
    print(f"   Total Transactions: {total}")
    print(f"   🚨 Flagged as FRAUD: {fraud_detected} ({fraud_detected/total*100:.1f}%)")
    print(f"   ✅ Flagged as LEGIT: {legit_detected} ({legit_detected/total*100:.1f}%)")
    print(f"   ❌ Errors: {errors}")
    
    # Check if ground truth exists
    if 'has_balance_mismatch' in audit.columns:
        print(f"\n🎯 Ground Truth Comparison:")
        actual_frauds = audit['has_balance_mismatch'].sum()
        print(f"   Actual Frauds (has_balance_mismatch=1): {actual_frauds}")
        
        # Confusion matrix
        if fraud_detected > 0 or actual_frauds > 0:
            tp = ((audit['Detection'] == 'FRAUD') & (audit['has_balance_mismatch'] == 1)).sum()
            fp = ((audit['Detection'] == 'FRAUD') & (audit['has_balance_mismatch'] == 0)).sum()
            fn = ((audit['Detection'] == 'LEGIT') & (audit['has_balance_mismatch'] == 1)).sum()
            tn = ((audit['Detection'] == 'LEGIT') & (audit['has_balance_mismatch'] == 0)).sum()
            
            print(f"\n   Confusion Matrix:")
            print(f"   ┌──────────────┬─────────┬─────────┐")
            print(f"   │              │ Pred F  │ Pred L  │")
            print(f"   ├──────────────┼─────────┼─────────┤")
            print(f"   │ Actual Fraud │  {tp:3d} TP │  {fn:3d} FN │")
            print(f"   │ Actual Legit │  {fp:3d} FP │  {tn:3d} TN │")
            print(f"   └──────────────┴─────────┴─────────┘")
            
            # Metrics
            if actual_frauds > 0:
                recall = tp / actual_frauds * 100
                print(f"\n   📈 Performance Metrics:")
                print(f"      Recall (Fraud Detection Rate): {recall:.1f}%")
                print(f"      Missed Frauds: {fn}")
            
            if fraud_detected > 0:
                precision = tp / fraud_detected * 100
                print(f"      Precision (Accuracy of Alerts): {precision:.1f}%")
                print(f"      False Alarms: {fp}")
    
    # Show fraud details
    if fraud_detected > 0:
        print(f"\n📋 Detected Fraud Transactions:")
        fraud_rows = audit[audit['Detection'] == 'FRAUD']
        for idx, row in fraud_rows.iterrows():
            amount = row.get('amount', 'N/A')
            error_orig = row.get('errorbalanceorig', 'N/A')
            error_dest = row.get('errorbalancedest', 'N/A')
            insight = row.get('Insight', 'No explanation')
            
            print(f"\n   Transaction #{idx}:")
            print(f"      Amount: ${amount:,.2f}" if isinstance(amount, (int, float)) else f"      Amount: {amount}")
            print(f"      Error Orig: {error_orig}")
            print(f"      Error Dest: {error_dest}")
            print(f"      Model: {insight[:80]}...")

print(f"\n{'='*80}")
print(" END OF REPORT")
print("="*80)
