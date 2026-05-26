"""
Test XGBoost model with new retrained version
"""
import sys
# Force UTF-8 encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import joblib
import numpy as np

print("=" * 80)
print(" TESTING RETRAINED XGBOOST MODEL")
print("="*80)

# Load demo data
try:
    demo = pd.read_csv('data_generation/creditcard_demo_final.csv')
except:
    # Fallback to creating a dummy row if file not found
    print("Demo file not found, creating dummy data...")
    cols = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 
            'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 
            'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26', 'v27', 'v28', 
            'amount_scaled', 'hour', 'dayofweek', 'is_weekend']
    demo = pd.DataFrame(np.random.randn(5, 32), columns=cols)
    demo['is_fraud'] = [0, 0, 0, 0, 1]

# Load models
print("Loading models...")
try:
    xgb = joblib.load('models/cc_xgb_model.pkl')
    rf = joblib.load('models/cc_rf_model.pkl')
    scaler = joblib.load('models/cc_scaler.pkl')
    features = joblib.load('models/cc_features.pkl')
    weights = np.load('models/cc_ensemble_weights.npy')
    threshold = np.load('models/cc_ensemble_threshold.npy')
    
    print(f"\n✅ All artifacts loaded successfully")
    print(f"   Features: {len(features)}")
    print(f"   Ensemble Weights: {weights}")
    print(f"   Ensemble Threshold: {threshold}")

    # Test on a few rows
    print("\nRunning predictions...")
    for idx in range(min(5, len(demo))):
        if 'is_fraud' in demo.columns:
            row = demo.iloc[idx].drop('is_fraud')
            is_fraud = demo.iloc[idx]['is_fraud']
            label = f"(is_fraud={is_fraud})"
        else:
            row = demo.iloc[idx]
            label = "(unknown label)"
        
        # Ensure correct features
        X = pd.DataFrame([row])
        # Rename columns to lowercase if needed
        X.columns = X.columns.str.lower()
        
        # Select features
        X = X[features]
        X_scaled = scaler.transform(X)
        
        prob_xgb = xgb.predict_proba(X_scaled)[0, 1]
        prob_rf = rf.predict_proba(X_scaled)[0, 1]
        prob_ensemble = weights[0]*prob_xgb + weights[1]*prob_rf
        
        print(f"\nRow {idx} {label}:")
        print(f"   XGB prob: {prob_xgb:.6f}")
        print(f"   RF prob:  {prob_rf:.6f}")
        print(f"   Ensemble: {prob_ensemble:.6f}")
        
        if prob_xgb > 0.001:
            print("   ✅ XGBoost is working! (returned non-zero probability)")
        else:
            print("   ⚠️ XGBoost returned very low probability (could be legit, or issue persists)")

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")

print("\n" + "="*80)
