import pandas as pd
from sklearn.model_selection import train_test_split
import os

DATA_PATH = "Fraud_Detection_Model_Paysim_CC/data/cleaned_paysim.csv"
OUTPUT_DIR = "Fraud_Detection_Model_Paysim_CC/data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["isFraud"],
    random_state=42
)

test_path = os.path.join(OUTPUT_DIR, "paysim_test.csv")
test_df.to_csv(test_path, index=False)

print(f"✅ PaySim test set created: {test_path}")
print(test_df["isFraud"].value_counts())
