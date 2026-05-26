"""
experiments/test_demo_csv.py
────────────────────────────
Verifies that the generated 'paysim_demo_v3.csv' is correctly processed 
by the V3 Hybrid Model backend.

Usage:
    python experiments/test_demo_csv.py
"""
import requests
import pandas as pd
import sys
import time

API_URL = "http://127.0.0.1:8003/predict"
DEMO_CSV = "experiments/paysim/data_generation/paysim_demo_v3.csv"

def test_demo_prediction():
    print(f"Loading {DEMO_CSV}...")
    try:
        df = pd.read_csv(DEMO_CSV)
    except FileNotFoundError:
        print(f"❌ Error: {DEMO_CSV} not found. Run generate_paysim_demo_csv.py first.")
        return

    print(f"Loaded {len(df)} rows. Columns: {len(df.columns)}")
    
    # Check features
    expected_cols = 18
    if len(df.columns) != expected_cols:
        print(f"⚠️ Warning: Expected {expected_cols} columns, got {len(df.columns)}")
    
    # Pick a random sample
    sample = df.sample(1).iloc[0]
    print("\nTest Transaction:")
    print(f"  Amount: {sample['amount']}")
    print(f"  Type: {'TRANSFER' if sample['upi_type_upi_transfer'] else 'PAYMENT'}")
    print(f"  Velocity (count): {sample['tx_count_cumul']}")
    
    payload = {
        "transaction_type": "paysim",
        "tabular_features": sample.tolist(),
        "lstm_sequence": None
    }
    
    print("\nSending to backend...")
    try:
        t0 = time.time()
        resp = requests.post(API_URL, json=payload, timeout=10)
        t1 = time.time()
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ Backend Response ({t1-t0:.3f}s):")
            print(f"  Decision: {data['decision']}")
            print(f"  Confidence: {data['confidence']:.4f}")
            print(f"  Explanation: {data['explanation']}")
            print(f"  Review Flag: {data['review_flag']}")
            
            if data['decision'] in ['BLOCK', 'REVIEW', 'ALLOW']:
                print("\n🎉 Verification PASSED: Backend accepted V3 data format.")
            else:
                print("\n❓ Verification AMBIGUOUS: Unknown decision.")
        else:
            print(f"❌ Backend Error: {resp.status_code}")
            print(resp.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ ConnectError: Is the backend running? (uvicorn backend.app:app ...)")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_demo_prediction()