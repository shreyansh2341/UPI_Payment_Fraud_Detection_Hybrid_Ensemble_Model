from final_ensemble_inference import predict_paysim, predict_creditcard
import joblib

# ---- PaySim dummy transaction ----
paysim_sample = {
    "amount": 5000,
    "oldbalanceOrg": 10000,
    "newbalanceOrig": 5000,
    "oldbalanceDest": 0,
    "newbalanceDest": 5000,
    "hour": 14,
    "dayofweek": 2,
    "is_weekend": 0,
    "errorBalanceOrig": 0,
    "errorBalanceDest": 0,
    "has_balance_mismatch": 1
}

fraud, score, reason = predict_paysim(paysim_sample)
print("PaySim →", fraud, score, reason)


# ---- Credit Card dummy transaction (FIXED) ----
cc_features = joblib.load(
    "Fraud_Detection_Model_Paysim_CC/models/cc_features.pkl"
)

cc_sample = {feature: 0.1 for feature in cc_features}

fraud, score, reason = predict_creditcard(cc_sample)
print("CreditCard →", fraud, score, reason)
