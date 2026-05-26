import pandas as pd

print("="*60)
print("DEMO DATASET FRAUD ANALYSIS")
print("="*60)

# Analyze paysim_demo_final.csv
print("\n📊 File: paysim_demo_final.csv")
print("-"*60)
try:
    df1 = pd.read_csv('data_generation/paysim_demo_final.csv')
    print(f"Total Transactions: {len(df1):,}")
    print(f"Columns: {list(df1.columns)}")
    
    if 'isFraud' in df1.columns:
        fraud_count = df1['isFraud'].sum()
        legit_count = len(df1) - fraud_count
        fraud_pct = (fraud_count / len(df1)) * 100
        
        print(f"\n✅ Legitimate: {legit_count:,} ({100-fraud_pct:.2f}%)")
        print(f"🚨 Fraud: {fraud_count:,} ({fraud_pct:.2f}%)")
        
        print(f"\nFraud Distribution:")
        print(df1['isFraud'].value_counts())
    else:
        print("⚠️ No 'isFraud' column found")
except Exception as e:
    print(f"❌ Error: {e}")

# Analyze paysim_demo.csv
print("\n" + "="*60)
print("\n📊 File: paysim_demo.csv")
print("-"*60)
try:
    df2 = pd.read_csv('data_generation/paysim_demo.csv')
    print(f"Total Transactions: {len(df2):,}")
    print(f"Columns: {list(df2.columns)}")
    
    if 'isFraud' in df2.columns:
        fraud_count = df2['isFraud'].sum()
        legit_count = len(df2) - fraud_count
        fraud_pct = (fraud_count / len(df2)) * 100
        
        print(f"\n✅ Legitimate: {legit_count:,} ({100-fraud_pct:.2f}%)")
        print(f"🚨 Fraud: {fraud_count:,} ({fraud_pct:.2f}%)")
        
        print(f"\nFraud Distribution:")
        print(df2['isFraud'].value_counts())
    else:
        print("⚠️ No 'isFraud' column found")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
