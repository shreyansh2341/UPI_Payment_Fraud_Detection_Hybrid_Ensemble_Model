import pandas as pd
import sys
sys.path.insert(0, '.')

from src.utils.preprocessor import clean_and_engineer_upi
from src.final_ensemble_inference import ensemble_predict

# Load demo CSV
df = pd.read_csv("data_generation/paysim_demo_final.csv")

# Test row 37
row = df.iloc[37:38]  # Select as DataFrame
row_eng = clean_and_engineer_upi(row)

# Add UPI type flags
row_eng['upi_type_upi_payment'] = 0
row_eng['upi_type_upi_transfer'] = 1

# Predict
result = ensemble_predict("paysim", row_eng, lstm_sequence=None)

print(f"Row 37 Prediction:")
print(f"  Decision: {result['decision']}")
print(f"  Score: {result['score']:.4f}")
print(f"  Explanation: {result['explanation']}")