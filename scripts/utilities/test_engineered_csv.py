"""Test script to verify inference on pre-engineered CSV format."""
import pandas as pd
import sys
sys.path.insert(0, '.')

from src.final_ensemble_inference import ensemble_predict

# Load demo CSV (pre-engineered format)
df = pd.read_csv('data_generation/paysim_demo_final.csv')
print(f'Columns: {list(df.columns)}')
print(f'Total rows: {len(df)}')

# Test first 5 rows
for i in range(min(5, len(df))):
    row = df.iloc[i:i+1].copy()
    
    # Normalize column names to lowercase (model expects lowercase)
    row = row.rename(columns={
        'oldbalanceOrg': 'oldbalanceorg',
        'newbalanceOrig': 'newbalanceorig',
        'oldbalanceDest': 'oldbalancedest',
        'newbalanceDest': 'newbalancedest',
        'errorBalanceOrig': 'errorbalanceorig',
        'errorBalanceDest': 'errorbalancedest',
    })
    
    # Add missing UPI type columns
    row['upi_type_upi_payment'] = 0
    row['upi_type_upi_transfer'] = 1
    
    result = ensemble_predict('paysim', row, lstm_sequence=None)
    decision = "FRAUD" if result['decision'] else "LEGIT"
    print(f'Row {i}: {decision}, Score={result["score"]:.4f}, {result["explanation"]}')
