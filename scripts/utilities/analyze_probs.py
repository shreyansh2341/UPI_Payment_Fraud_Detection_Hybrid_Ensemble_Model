# Analyze fraud probabilities and set threshold
import pandas as pd
import numpy as np
import joblib

demo = pd.read_csv('data_generation/creditcard_demo_final.csv')
xgb = joblib.load('models/cc_xgb_model.pkl')
rf = joblib.load('models/cc_rf_model.pkl')
scaler = joblib.load('models/cc_scaler.pkl')
features = joblib.load('models/cc_features.pkl')
weights = np.load('models/cc_ensemble_weights.npy')

print("="*60)
print("FRAUD PROBABILITIES")
print("="*60)

fraud_probs = []
legit_probs = []

for idx in range(len(demo)):
    row = demo.iloc[idx]
    is_fraud = row['is_fraud']
    X = pd.DataFrame([row.drop('is_fraud')])
    X.columns = X.columns.str.lower()
    X = X[features]
    X_scaled = scaler.transform(X)
    
    prob_xgb = xgb.predict_proba(X_scaled)[0, 1]
    prob_rf = rf.predict_proba(X_scaled)[0, 1]
    prob_ensemble = weights[0]*prob_xgb + weights[1]*prob_rf
    
    if is_fraud == 1:
        fraud_probs.append(prob_ensemble)
        print(f"FRAUD idx={idx}: Ens={prob_ensemble:.4f} XGB={prob_xgb:.4f} RF={prob_rf:.4f}")
    else:
        legit_probs.append(prob_ensemble)

print()
print(f"Fraud prob range: {min(fraud_probs):.4f} - {max(fraud_probs):.4f}")
print(f"Legit prob range: {min(legit_probs):.4f} - {max(legit_probs):.4f}")
print(f"Legit 95th pct:   {np.percentile(legit_probs, 95):.4f}")

# Set threshold just below minimum fraud probability
new_threshold = min(fraud_probs) * 0.95
print(f"\nNew threshold: {new_threshold:.6f}")

np.save('models/cc_ensemble_threshold.npy', np.array([new_threshold]))
print("Saved!")
