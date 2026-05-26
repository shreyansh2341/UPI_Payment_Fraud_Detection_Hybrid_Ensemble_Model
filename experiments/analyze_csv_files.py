"""
analyze_csv_files.py — Analyze test CSV files through V5 Hybrid Pipeline
========================================================================
Runs all csv_files/*.csv through the V5 batch inference engine and
compares predicted 4-tier decisions against ground-truth isFraud labels.
"""
import sys, os, glob
import pandas as pd
import numpy as np

sys.path.insert(0, r"g:\My Drive\Fraud_Detection_Model")
os.chdir(r"g:\My Drive\Fraud_Detection_Model")

from backend.inference import run_v5_batch_inference

csv_dir = r"g:\My Drive\Fraud_Detection_Model\csv_files"
csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))

for file in csv_files:
    fname = os.path.basename(file)
    df = pd.read_csv(file)
    n = len(df)

    # Ground truth
    has_gt = "isFraud" in df.columns
    gt_fraud = int(df["isFraud"].sum()) if has_gt else "N/A"
    gt_legit = int(n - df["isFraud"].sum()) if has_gt else "N/A"

    # Run V5
    csv_data = df.values.tolist()
    csv_columns = df.columns.tolist()
    results = run_v5_batch_inference(csv_data, csv_columns, "paysim")

    # Tally 4-tier decisions
    tiers = {"BLOCK": 0, "BLOCK_NOVEL": 0, "REVIEW": 0, "ALLOW": 0}
    for r in results:
        label = r.get("decision_label", "ALLOW")
        tiers[label] = tiers.get(label, 0) + 1

    print(f"\n{'='*60}")
    print(f"FILE: {fname}  (Total rows: {n})")
    print(f"{'='*60}")
    print(f"Ground Truth:  Fraud={gt_fraud},  Legit={gt_legit}")
    print(f"")
    print(f"V5 4-Tier Decision Distribution:")
    print(f"  [BLOCK] Known Fraud:       {tiers['BLOCK']}")
    print(f"  [BLOCK_NOVEL] Novel Fraud: {tiers['BLOCK_NOVEL']}")
    print(f"  [REVIEW] Suspicious:       {tiers['REVIEW']}")
    print(f"  [ALLOW] Legitimate:        {tiers['ALLOW']}")

    # Confusion analysis if ground truth available
    if has_gt:
        gt = df["isFraud"].values
        pred_fraud = np.array([
            1 if r.get("decision_label") in ("BLOCK", "BLOCK_NOVEL") else 0
            for r in results
        ])

        tp = int(((pred_fraud == 1) & (gt == 1)).sum())
        fp = int(((pred_fraud == 1) & (gt == 0)).sum())
        fn = int(((pred_fraud == 0) & (gt == 1)).sum())
        tn = int(((pred_fraud == 0) & (gt == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"")
        print(f"  Confusion Matrix (BLOCK+BLOCK_NOVEL vs isFraud):")
        print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"    Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        # Review analysis
        review_on_fraud = sum(
            1 for i in range(n)
            if results[i].get("decision_label") == "REVIEW" and gt[i] == 1
        )
        review_on_legit = sum(
            1 for i in range(n)
            if results[i].get("decision_label") == "REVIEW" and gt[i] == 0
        )
        print(f"    REVIEW on actual fraud: {review_on_fraud}")
        print(f"    REVIEW on actual legit: {review_on_legit}")

        # Effective recall (BLOCK + BLOCK_NOVEL + REVIEW catching fraud)
        effective_catch = tp + review_on_fraud
        effective_recall = effective_catch / (tp + fn + review_on_fraud) if (tp + fn + review_on_fraud) > 0 else 0
        print(f"    Effective fraud catch (BLOCK+NOVEL+REVIEW): {effective_catch}/{gt_fraud}")

print(f"\n{'='*60}")
print("ANALYSIS COMPLETE")
print(f"{'='*60}")
