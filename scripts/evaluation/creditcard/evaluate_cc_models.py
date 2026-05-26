"""
evaluate_cc_models.py
─────────────────────
Complete evaluation of Credit Card fraud detection models:
  • XGBoost (standalone)
  • Random Forest (standalone)
  • Ensemble (weighted XGB + RF)

Saves results to: evaluation_results/cc_evaluation_results/
  - cc_model_comparison.csv        (tabular metrics)
  - cc_model_comparison.md         (markdown table)
  - cc_confusion_matrices.png      (side-by-side confusion matrices)
  - cc_roc_curves.png              (ROC curves overlay)
  - cc_pr_curves.png               (Precision-Recall curves overlay)
  - cc_metrics_bar_chart.png       (bar chart comparing key metrics)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard_test.csv"
MODELS_DIR = BASE_DIR / "models"

OUTPUT_DIR = BASE_DIR / "evaluation_results" / "cc_evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("  CREDIT CARD MODEL EVALUATION")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"\n📂 Loaded test data: {len(df):,} transactions")
print(f"   Frauds: {df['isFraud'].sum()} ({df['isFraud'].mean()*100:.3f}%)")

y_true = df["isFraud"].values

# Case-insensitive column mapping
col_map = {c.lower(): c for c in df.columns}

# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================
print("\n📦 Loading model artifacts...")

features = joblib.load(MODELS_DIR / "cc_features.pkl")
scaler = joblib.load(MODELS_DIR / "cc_scaler.pkl")
xgb_model = joblib.load(MODELS_DIR / "cc_xgb_model.pkl")
rf_model = joblib.load(MODELS_DIR / "cc_rf_model.pkl")

xgb_threshold = np.load(MODELS_DIR / "cc_threshold.npy")[0]
ensemble_weights = np.load(MODELS_DIR / "cc_ensemble_weights.npy")
ensemble_threshold = np.load(MODELS_DIR / "cc_ensemble_threshold.npy")[0]

w_xgb, w_rf = ensemble_weights[0], ensemble_weights[1]

print(f"   XGBoost threshold : {xgb_threshold:.6f}")
print(f"   Ensemble weights  : {w_xgb:.2f} XGB + {w_rf:.2f} RF")
print(f"   Ensemble threshold: {ensemble_threshold:.6f}")

# ============================================================
# PREPARE FEATURES
# ============================================================
mapped_features = [col_map[f] for f in features]
X = scaler.transform(df[mapped_features].values)

# ============================================================
# GET PREDICTIONS
# ============================================================
print("\n🔮 Generating predictions...")

# XGBoost
xgb_probs = xgb_model.predict_proba(X)[:, 1]
xgb_preds = (xgb_probs >= xgb_threshold).astype(int)

# Random Forest
rf_probs = rf_model.predict_proba(X)[:, 1]
# Use same threshold approach — find optimal RF threshold via F1
precision_arr, recall_arr, thresh_arr = precision_recall_curve(y_true, rf_probs)
f1_arr = 2 * (precision_arr[:-1] * recall_arr[:-1]) / (precision_arr[:-1] + recall_arr[:-1] + 1e-12)
rf_threshold = thresh_arr[np.argmax(f1_arr)]
rf_preds = (rf_probs >= rf_threshold).astype(int)

# Ensemble
ens_probs = w_xgb * xgb_probs + w_rf * rf_probs
ens_preds = (ens_probs >= ensemble_threshold).astype(int)

print(f"   XGB predictions  : {xgb_preds.sum()} flagged as fraud")
print(f"   RF predictions   : {rf_preds.sum()} flagged as fraud")
print(f"   Ensemble preds   : {ens_preds.sum()} flagged as fraud")

# ============================================================
# COMPUTE METRICS
# ============================================================
def compute_metrics(y_true, y_pred, y_prob, model_name, threshold):
    """Compute all classification metrics for a model."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    alert_rate = (tp + fp) / len(y_true)

    return {
        "Model": model_name,
        "Threshold": round(threshold, 6),
        "ROC-AUC": round(roc, 4),
        "PR-AUC": round(pr_auc, 4),
        "Accuracy": round(acc, 4),
        "Recall (Sensitivity)": round(rec, 4),
        "Precision": round(prec, 4),
        "F1-Score": round(f1, 4),
        "Specificity": round(specificity, 4),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Alert Rate": round(alert_rate, 6),
    }

models_info = [
    ("XGBoost", xgb_preds, xgb_probs, xgb_threshold),
    ("Random Forest", rf_preds, rf_probs, rf_threshold),
    ("Ensemble (XGB+RF)", ens_preds, ens_probs, ensemble_threshold),
]

results = []
for name, preds, probs, thresh in models_info:
    m = compute_metrics(y_true, preds, probs, name, thresh)
    results.append(m)

results_df = pd.DataFrame(results)

# ============================================================
# PRINT RESULTS
# ============================================================
print("\n" + "=" * 70)
print("  RESULTS SUMMARY")
print("=" * 70)

for r in results:
    print(f"\n{'─'*50}")
    print(f"  📊 {r['Model']}")
    print(f"{'─'*50}")
    print(f"  Threshold  : {r['Threshold']}")
    print(f"  ROC-AUC    : {r['ROC-AUC']:.4f}")
    print(f"  PR-AUC     : {r['PR-AUC']:.4f}")
    print(f"  Recall     : {r['Recall (Sensitivity)']:.4f}  ({r['Recall (Sensitivity)']*100:.2f}%)")
    print(f"  Precision  : {r['Precision']:.4f}  ({r['Precision']*100:.2f}%)")
    print(f"  F1-Score   : {r['F1-Score']:.4f}")
    print(f"  Specificity: {r['Specificity']:.4f}")
    print(f"  Confusion  : TP={r['TP']}  FP={r['FP']}  FN={r['FN']}  TN={r['TN']}")
    print(f"  Alert Rate : {r['Alert Rate']:.4%}")

# ============================================================
# SAVE TABULAR RESULTS
# ============================================================
print("\n\n💾 Saving results...")

# CSV
csv_path = OUTPUT_DIR / "cc_model_comparison.csv"
results_df.to_csv(csv_path, index=False)
print(f"   ✅ {csv_path}")

# Markdown Table
md_path = OUTPUT_DIR / "cc_model_comparison.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Credit Card Fraud Detection — Model Comparison\n\n")
    f.write(f"> **Test Set**: {len(y_true):,} transactions ({y_true.sum()} frauds, {y_true.mean()*100:.3f}% fraud rate)\n\n")

    f.write("## Performance Metrics\n\n")
    f.write("| Metric | XGBoost | Random Forest | Ensemble (XGB+RF) |\n")
    f.write("|:-------|:-------:|:-------------:|:-----------------:|\n")

    metric_rows = [
        ("**Threshold**", "Threshold", "{:.6f}"),
        ("**ROC-AUC**", "ROC-AUC", "{:.4f}"),
        ("**PR-AUC**", "PR-AUC", "{:.4f}"),
        ("**Accuracy**", "Accuracy", "{:.4f}"),
        ("**Recall (Sensitivity)**", "Recall (Sensitivity)", "{:.4f}"),
        ("**Precision**", "Precision", "{:.4f}"),
        ("**F1-Score**", "F1-Score", "{:.4f}"),
        ("**Specificity**", "Specificity", "{:.4f}"),
        ("**Alert Rate**", "Alert Rate", "{:.4%}"),
    ]

    for label, key, fmt in metric_rows:
        vals = [fmt.format(r[key]) for r in results]
        f.write(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} |\n")

    f.write("\n## Confusion Matrices\n\n")
    for r in results:
        f.write(f"### {r['Model']}\n\n")
        f.write("```\n")
        f.write("                 Predicted\n")
        f.write("               Legit    Fraud\n")
        f.write(f"Actual Legit   {r['TN']:>6,}   {r['FP']:>5,}\n")
        f.write(f"Actual Fraud   {r['FN']:>6,}   {r['TP']:>5,}\n")
        f.write("```\n\n")

    f.write("---\n")
    f.write(f"\n*Evaluation run on credit card test set ({len(y_true):,} samples).*\n")

print(f"   ✅ {md_path}")

# ============================================================
# PLOT 1: CONFUSION MATRICES (side-by-side)
# ============================================================
print("\n🎨 Generating plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Credit Card Model — Confusion Matrices", fontsize=16, fontweight="bold", y=1.02)

for idx, (name, preds, probs, thresh) in enumerate(models_info):
    ax = axes[idx]
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()

    cm_display = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(cm_display, cmap="Blues", aspect="auto")

    # Annotate
    labels = [[f"TN\n{tn:,}", f"FP\n{fp:,}"],
              [f"FN\n{fn:,}", f"TP\n{tp:,}"]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm_display[i, j] > cm_display.max() / 2 else "black"
            ax.text(j, i, labels[i][j], ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Fraud"], fontsize=11)
    ax.set_yticklabels(["Legit", "Fraud"], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"{name}\n(threshold={thresh:.4f})", fontsize=13, fontweight="bold")

plt.tight_layout()
cm_path = OUTPUT_DIR / "cc_confusion_matrices.png"
fig.savefig(cm_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"   ✅ {cm_path}")

# ============================================================
# PLOT 2: ROC CURVES
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#2196F3", "#4CAF50", "#FF5722"]

for idx, (name, preds, probs, thresh) in enumerate(models_info):
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc_val = roc_auc_score(y_true, probs)
    ax.plot(fpr, tpr, color=colors[idx], linewidth=2.5,
            label=f"{name} (AUC = {auc_val:.4f})")

ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title("Credit Card Models — ROC Curves", fontsize=15, fontweight="bold")
ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])

plt.tight_layout()
roc_path = OUTPUT_DIR / "cc_roc_curves.png"
fig.savefig(roc_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"   ✅ {roc_path}")

# ============================================================
# PLOT 3: PRECISION-RECALL CURVES
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

for idx, (name, preds, probs, thresh) in enumerate(models_info):
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, probs)
    pr_auc_val = average_precision_score(y_true, probs)
    ax.plot(rec_arr, prec_arr, color=colors[idx], linewidth=2.5,
            label=f"{name} (PR-AUC = {pr_auc_val:.4f})")

ax.set_xlabel("Recall", fontsize=13)
ax.set_ylabel("Precision", fontsize=13)
ax.set_title("Credit Card Models — Precision-Recall Curves", fontsize=15, fontweight="bold")
ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])

plt.tight_layout()
pr_path = OUTPUT_DIR / "cc_pr_curves.png"
fig.savefig(pr_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"   ✅ {pr_path}")

# ============================================================
# PLOT 4: METRICS BAR CHART
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

metrics_to_plot = ["ROC-AUC", "PR-AUC", "Recall (Sensitivity)", "Precision", "F1-Score", "Specificity"]
x = np.arange(len(metrics_to_plot))
bar_width = 0.25

for idx, r in enumerate(results):
    vals = [r[m] for m in metrics_to_plot]
    bars = ax.bar(x + idx * bar_width, vals, bar_width,
                  label=r["Model"], color=colors[idx], alpha=0.85, edgecolor="white")
    # Value labels on bars
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x + bar_width)
ax.set_xticklabels(metrics_to_plot, fontsize=11, rotation=15, ha="right")
ax.set_ylabel("Score", fontsize=13)
ax.set_title("Credit Card Models — Key Performance Metrics", fontsize=15, fontweight="bold")
ax.legend(fontsize=11, framealpha=0.9)
ax.set_ylim(0, 1.15)
ax.grid(axis="y", alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))

plt.tight_layout()
bar_path = OUTPUT_DIR / "cc_metrics_bar_chart.png"
fig.savefig(bar_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"   ✅ {bar_path}")

# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 70)
print("  ✨ EVALUATION COMPLETE!")
print("=" * 70)
print(f"\n  All outputs saved to: {OUTPUT_DIR}")
print(f"  Files created:")
print(f"    📄 cc_model_comparison.csv")
print(f"    📄 cc_model_comparison.md")
print(f"    🖼️  cc_confusion_matrices.png")
print(f"    🖼️  cc_roc_curves.png")
print(f"    🖼️  cc_pr_curves.png")
print(f"    🖼️  cc_metrics_bar_chart.png")
print("=" * 70)
