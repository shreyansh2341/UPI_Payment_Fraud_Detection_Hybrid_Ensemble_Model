import numpy as np
from final_ensemble_inference import ensemble_predict

# ---- Dummy inputs (just to test flow) ----
dummy_lstm_seq = np.zeros((1, 5, 12))      # PaySim LSTM shape
dummy_tabular = np.zeros((1, 14))          # PaySim tabular features

decision, reason = ensemble_predict(
    transaction_type="paysim",
    tabular_features=dummy_tabular,
    lstm_sequence=dummy_lstm_seq
)

print("Decision:", "FRAUD" if decision else "LEGIT")
print("Reason  :", reason)
