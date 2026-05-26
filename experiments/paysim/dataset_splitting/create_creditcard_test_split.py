import pandas as pd
from sklearn.model_selection import train_test_split
import os

DATA_PATH = "Fraud_Detection_Model_Paysim_CC/data/cleaned_creditcard.csv"
OUTPUT_DIR = "Fraud_Detection_Model_Paysim_CC/data"

df = pd.read_csv(DATA_PATH)

print("Columns found:", df.columns.tolist())

# 🔴 CHANGE THIS after checking printed columns
TARGET_COL = "isFraud"   # <-- adjust if needed

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df[TARGET_COL],
    random_state=42
)

test_path = os.path.join(OUTPUT_DIR, "creditcard_test.csv")
test_df.to_csv(test_path, index=False)

print(f"\n✅ Credit Card test set created: {test_path}")
print("Test fraud distribution:")
print(test_df[TARGET_COL].value_counts())
