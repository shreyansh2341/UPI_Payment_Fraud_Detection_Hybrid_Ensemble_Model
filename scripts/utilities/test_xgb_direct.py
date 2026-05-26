"""Test XGBoost model directly"""
import pandas as pd
import joblib
import numpy as np

print("=" * 80)
print(" TESTING XGBOOST MODEL DIRECTLY")
print("="*80)

# Load demo data
demo = pd.read_csv('data_generation/creditcard_demo_final.csv')

# Load models
xgb = joblib.load('models/cc_xgb_model.pkl')
rf = joblib.load('models/cc_rf_model.pkl')
scaler = joblib.load('models/cc_scaler.pkl')
features = joblib.load('models/cc_features.pkl')

print(f"\n✅ Models loaded")
print(f"   Features: {len(features)}")
print(f"   First 5 features: {features[:5]}")

# Test on a few rows
for idx in [0, 5, 15, 38, 39]:
    row = demo.iloc[idx].drop('is_fraud')
    is_fraud = demo.iloc[idx]['is_fraud']
    
    X = pd.DataFrame([row])[features]
    X_scaled = scaler.transform(X)
    
    prob_xgb = xgb.predict_proba(X_scaled)[0, 1]
    prob_rf = rf.predict_proba(X_scaled)[0, 1]
    
    print(f"\nRow {idx} (is_fraud={is_fraud}):")
    print(f"   XGB prob: {prob_xgb:.6f}")
    print(f"   RF prob:  {prob_rf:.6f}")
    print(f"   Ensemble: {0.5*prob_xgb + 0.5*prob_rf:.6f}")

print("\n" + "="*80)
