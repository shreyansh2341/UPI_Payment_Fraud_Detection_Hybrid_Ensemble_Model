import pandas as pd
from final_ensemble_inference import (
    predict_paysim_batch,
    predict_creditcard_batch
)

def batch_predict(
    input_csv,
    output_csv,
    dataset_type,
    chunk_size=20000
):
    reader = pd.read_csv(input_csv, chunksize=chunk_size)

    first_chunk = True

    for i, df in enumerate(reader, start=1):
        print(f"🚀 Processing chunk {i} ({len(df)} rows)")

        if dataset_type == "paysim":
            fraud, scores, reasons = predict_paysim_batch(df)
        elif dataset_type == "creditcard":
            fraud, scores, reasons = predict_creditcard_batch(df)
        else:
            raise ValueError("dataset_type must be 'paysim' or 'creditcard'")

        df["fraud_prediction"] = fraud
        df["fraud_score"] = scores
        df["risk_level"] = reasons

        # append to CSV incrementally
        df.to_csv(
            output_csv,
            mode="w" if first_chunk else "a",
            index=False,
            header=first_chunk
        )

        first_chunk = False

    print(f"\n✅ Batch prediction completed → {output_csv}")


# =======================
# 🚀 ENTRY POINT
# =======================
if __name__ == "__main__":

    INPUT_CSV = "Fraud_Detection_Model_Paysim_CC/data/creditcard_test.csv"
    OUTPUT_CSV = "Fraud_Detection_Model_Paysim_CC/data/creditcard_predictions.csv"
    DATASET_TYPE = "creditcard"   # or "paysim"

    batch_predict(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        dataset_type=DATASET_TYPE,
        chunk_size=20000
    )
