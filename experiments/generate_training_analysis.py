"""
generate_training_analysis.py — V3 Hybrid Training Analysis
═══════════════════════════════════════════════════════════════
Trains the V3 Hybrid model at different training data percentages
(40%–80% of the 70% training split). Tracks per-epoch AE loss and
computes full metrics (Accuracy, Precision, Recall, F1, PR-AUC) for
each model variant (AE standalone, RF, XGBoost, V3 Hybrid Ensemble).

Outputs:
  1. experiments/results/training_analysis_results.json  — raw data
  2. experiments/results/v3_training_analysis_epochs.png — composite chart
"""
import numpy as np
import pandas as pd
import joblib
import os
import gc
import json
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, precision_recall_curve, auc
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
DATA_PATH = "data/cleaned_paysim_lstm.csv"
RESULTS_DIR = "experiments/results"

DATA_PERCENTAGES = [0.40, 0.50, 0.60, 0.70, 0.80]
AE_EPOCHS = 20
AE_BATCH = 1024
AE_PATIENCE = 3
BLOCK_THRESHOLD = 0.77
AE_REVIEW_THRESHOLD_PERCENTILE = 95

os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════
def ae_error(model, X):
    """Compute log-scaled MSE reconstruction error."""
    recon = model.predict(X, batch_size=2048, verbose=0)
    err = np.mean(np.square(X - recon), axis=1)
    return np.log1p(err)

def calc_pr_auc(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

def full_metrics(y_true, y_pred, y_prob):
    """Compute a full dictionary of classification metrics."""
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score":  float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc":    float(calc_pr_auc(y_true, y_prob)),
    }

# ══════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════
print("═" * 70)
print("  V3 HYBRID — TRAINING ANALYSIS ACROSS PERCENTAGES & EPOCHS")
print("═" * 70)

print("\nLoading PaySim dataset ...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower()
df = df.sort_values("step").reset_index(drop=True)

TARGET = "isfraud"
DROP_COLS = ["isfraud", "nameorig", "namedest", "step", "datetime"]
FEATURES = [c for c in df.columns if c not in DROP_COLS]

X = df[FEATURES].values
y = df[TARGET].values

n = len(df)
train_end_max = int(0.70 * n)
val_end       = int(0.85 * n)

X_test, y_test = X[val_end:], y[val_end:]
print(f"  Total rows:       {n:,}")
print(f"  Max training:     {train_end_max:,} (70%)")
print(f"  Fixed test set:   {len(y_test):,} (15%)")
print(f"  Test fraud ratio: {y_test.mean():.4f}")

del df
gc.collect()

# ══════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════
MODEL_NAMES = [
    "Autoencoder (Standalone)",
    "Random Forest",
    "XGBoost",
    "V3 Hybrid Ensemble",
]

all_results = {
    "percentages": [int(p * 100) for p in DATA_PERCENTAGES],
    "ae_epochs_count": AE_EPOCHS,
    "epoch_histories": {},       # pct -> {train_loss: [...], val_loss: [...]}
    "best_ae_epoch": {},         # pct -> epoch number (1-based)
    "model_metrics": {},         # pct -> {model_name -> {accuracy, precision, ...}}
}

for pct in DATA_PERCENTAGES:
    pct_label = f"{int(pct * 100)}%"
    print(f"\n{'─' * 60}")
    print(f"  TRAINING ON {pct_label} OF ASSIGNED TRAINING DATA")
    print(f"{'─' * 60}")

    current_train_end = int(train_end_max * pct)
    X_train_sub = X[:current_train_end]
    y_train_sub = y[:current_train_end]

    n_fraud = int(y_train_sub.sum())
    n_legit = int(len(y_train_sub) - n_fraud)
    print(f"  Training rows:  {len(X_train_sub):,}  (fraud={n_fraud:,}, legit={n_legit:,})")

    # ── Scale ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sub)
    X_test_scaled  = scaler.transform(X_test)

    X_train_normal = X_train_scaled[y_train_sub == 0]
    input_dim = X_train_normal.shape[1]

    # ──────────────────────────────────────────
    # 1. AUTOENCODER (per-epoch tracking)
    # ──────────────────────────────────────────
    print("  → Autoencoder (per-epoch tracking) ...")
    inp = Input(shape=(input_dim,))
    x = Dense(64, activation="relu")(inp)
    x = Dense(32, activation="relu")(x)
    encoded = Dense(16, activation="relu")(x)
    x = Dense(32, activation="relu")(encoded)
    x = Dense(64, activation="relu")(x)
    out = Dense(input_dim, activation="linear")(x)

    ae = Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")

    history = ae.fit(
        X_train_normal, X_train_normal,
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH,
        validation_split=0.1,
        shuffle=True,
        callbacks=[EarlyStopping(patience=AE_PATIENCE, restore_best_weights=True)],
        verbose=0,
    )

    train_loss = [float(v) for v in history.history["loss"]]
    val_loss   = [float(v) for v in history.history["val_loss"]]
    actual_epochs = len(train_loss)
    best_epoch = int(np.argmin(val_loss)) + 1  # 1-based

    all_results["epoch_histories"][pct_label] = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "actual_epochs": actual_epochs,
    }
    all_results["best_ae_epoch"][pct_label] = best_epoch

    print(f"    Epochs run:  {actual_epochs}/{AE_EPOCHS}")
    print(f"    Best epoch:  {best_epoch} (val_loss={val_loss[best_epoch-1]:.6f})")

    # AE reconstruction error
    ae_train_err = ae_error(ae, X_train_scaled)
    ae_test_err  = ae_error(ae, X_test_scaled)

    # AE standalone: threshold at 95th percentile of normal test errors
    ae_normal_test = ae_test_err[y_test == 0]
    ae_thresh = float(np.percentile(ae_normal_test, AE_REVIEW_THRESHOLD_PERCENTILE))

    ae_pred = (ae_test_err >= ae_thresh).astype(int)
    ae_met = full_metrics(y_test, ae_pred, ae_test_err)
    print(f"    AE standalone — Recall={ae_met['recall']:.3f}  Prec={ae_met['precision']:.3f}  F1={ae_met['f1_score']:.3f}")

    # ──────────────────────────────────────────
    # 2. SUPERVISED DATA (19 features)
    # ──────────────────────────────────────────
    X_train_sup = np.column_stack([X_train_scaled, ae_train_err])
    X_test_sup  = np.column_stack([X_test_scaled,  ae_test_err])
    scale_pos = float((y_train_sub == 0).sum()) / max(1, float((y_train_sub == 1).sum()))

    # ──────────────────────────────────────────
    # 3. RANDOM FOREST (19 features)
    # ──────────────────────────────────────────
    print("  → Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=12,
        class_weight="balanced", n_jobs=-1, random_state=42,
        max_samples=min(500_000, len(X_train_sup)),
    )
    rf.fit(X_train_sup, y_train_sub)
    gc.collect()
    rf_prob = rf.predict_proba(X_test_sup)[:, 1]
    rf_pred = (rf_prob >= 0.5).astype(int)
    rf_met = full_metrics(y_test, rf_pred, rf_prob)
    print(f"    RF — Recall={rf_met['recall']:.3f}  Prec={rf_met['precision']:.3f}  F1={rf_met['f1_score']:.3f}")

    # ──────────────────────────────────────────
    # 4. XGBOOST (19 features)
    # ──────────────────────────────────────────
    gc.collect()
    print("  → XGBoost ...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6,
        learning_rate=0.05, subsample=0.5,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="aucpr", tree_method="hist",
        n_jobs=-1, random_state=42,
    )
    xgb_model.fit(X_train_sup, y_train_sub)
    gc.collect()
    xgb_prob = xgb_model.predict_proba(X_test_sup)[:, 1]
    xgb_pred = (xgb_prob >= 0.5).astype(int)
    xgb_met = full_metrics(y_test, xgb_pred, xgb_prob)
    print(f"    XGB — Recall={xgb_met['recall']:.3f}  Prec={xgb_met['precision']:.3f}  F1={xgb_met['f1_score']:.3f}")

    # ──────────────────────────────────────────
    # 5. V3 HYBRID ENSEMBLE
    # ──────────────────────────────────────────
    print("  → V3 Hybrid Ensemble ...")
    ensemble_prob = 0.5 * xgb_prob + 0.5 * rf_prob
    ensemble_block = (ensemble_prob >= BLOCK_THRESHOLD).astype(int)

    # Path B: Isolation Forest
    iforest = IsolationForest(
        n_estimators=100, contamination=0.001, random_state=42,
        max_samples=min(10000, int((y_train_sub == 0).sum())),
        n_jobs=-1,
    )
    iforest.fit(X_train_scaled[y_train_sub == 0])
    iforest_pred = iforest.predict(X_test_scaled)  # -1 = anomaly

    ae_flag = ae_test_err >= ae_thresh
    iforest_flag = iforest_pred == -1
    review_flag = (ae_flag | iforest_flag) & (ensemble_block == 0)

    # Combined decision: BLOCK or REVIEW considered fraud detection
    hybrid_pred = (ensemble_block | review_flag.astype(int)).astype(int)
    hybrid_met = full_metrics(y_test, hybrid_pred, ensemble_prob)
    print(f"    V3 Hybrid — Recall={hybrid_met['recall']:.3f}  Prec={hybrid_met['precision']:.3f}  F1={hybrid_met['f1_score']:.3f}")

    # Store all results
    all_results["model_metrics"][pct_label] = {
        "Autoencoder (Standalone)": ae_met,
        "Random Forest": rf_met,
        "XGBoost": xgb_met,
        "V3 Hybrid Ensemble": hybrid_met,
    }

    # Cleanup
    del ae, rf, xgb_model, iforest
    del X_train_sup, X_test_sup, ae_train_err, ae_test_err
    del rf_prob, xgb_prob, ensemble_prob
    tf.keras.backend.clear_session()
    gc.collect()

# ══════════════════════════════════════════════════════════
# SAVE RESULTS JSON
# ══════════════════════════════════════════════════════════
results_path = os.path.join(RESULTS_DIR, "training_analysis_results.json")
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n✅ Results saved to: {results_path}")

# ══════════════════════════════════════════════════════════
# GENERATE COMPOSITE TRAINING ANALYSIS CHART
# ══════════════════════════════════════════════════════════
print("\n📊 Generating V3 Training Analysis composite chart ...")

# ── COLOR PALETTE ──
BG_COLOR      = "#0d1117"
CARD_COLOR    = "#161b22"
GRID_COLOR    = "#21262d"
TITLE_COLOR   = "#f0f6fc"
LABEL_COLOR   = "#c9d1d9"
ACCENT_COLORS = {
    "40%": "#f97583",  # coral-red
    "50%": "#d29922",  # amber
    "60%": "#56d364",  # green
    "70%": "#58a6ff",  # blue (best)
    "80%": "#bc8cff",  # purple
}

MODEL_COLORS = {
    "Autoencoder (Standalone)": "#f97583",
    "Random Forest":            "#56d364",
    "XGBoost":                  "#d29922",
    "V3 Hybrid Ensemble":       "#58a6ff",
}

fig = plt.figure(figsize=(24, 14), facecolor=BG_COLOR)
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.32, wspace=0.25,
                       left=0.06, right=0.96, top=0.90, bottom=0.07)

# ─── SUPTITLE ───
fig.suptitle(
    "V3 Hybrid Model — Training Analysis Across Epochs & Data Percentages",
    fontsize=22, fontweight="bold", color=TITLE_COLOR, y=0.97,
)
fig.text(0.5, 0.935,
    f"Autoencoder: max {AE_EPOCHS} epochs (EarlyStopping patience={AE_PATIENCE})  •  "
    f"Block threshold: {BLOCK_THRESHOLD}  •  Test set: {len(y_test):,} rows",
    ha="center", fontsize=12, color=LABEL_COLOR, style="italic")

# ═══════════════════════════════════════════
# PANEL 1 — AE Training Loss per Epoch
# ═══════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(CARD_COLOR)

for pct_label in all_results["epoch_histories"]:
    h = all_results["epoch_histories"][pct_label]
    epochs = range(1, h["actual_epochs"] + 1)
    color = ACCENT_COLORS[pct_label]
    best_ep = all_results["best_ae_epoch"][pct_label]

    ax1.plot(epochs, h["train_loss"], color=color, linewidth=2,
             label=f"{pct_label} train", alpha=0.85)
    ax1.plot(epochs, h["val_loss"], color=color, linewidth=2,
             linestyle="--", alpha=0.55)
    # Mark best epoch
    ax1.scatter([best_ep], [h["val_loss"][best_ep - 1]],
                color=color, s=100, zorder=5, edgecolors="white", linewidth=1.5)
    ax1.annotate(f"E{best_ep}", (best_ep, h["val_loss"][best_ep - 1]),
                 textcoords="offset points", xytext=(6, 8),
                 fontsize=9, fontweight="bold", color=color)

ax1.set_title("Autoencoder Training Loss per Epoch", fontsize=14,
              fontweight="bold", color=TITLE_COLOR, pad=12)
ax1.set_xlabel("Epoch", fontsize=12, color=LABEL_COLOR)
ax1.set_ylabel("MSE Loss", fontsize=12, color=LABEL_COLOR)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.tick_params(colors=LABEL_COLOR)
ax1.grid(True, color=GRID_COLOR, alpha=0.5, linestyle="--")
ax1.legend(fontsize=9, loc="upper right", frameon=True,
           facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=LABEL_COLOR,
           ncol=2)

# ═══════════════════════════════════════════
# PANEL 2 — AE Validation Loss per Epoch
# ═══════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(CARD_COLOR)

for pct_label in all_results["epoch_histories"]:
    h = all_results["epoch_histories"][pct_label]
    epochs = range(1, h["actual_epochs"] + 1)
    color = ACCENT_COLORS[pct_label]
    best_ep = all_results["best_ae_epoch"][pct_label]

    ax2.plot(epochs, h["val_loss"], color=color, linewidth=2.5,
             label=f"{pct_label}  (best E{best_ep})", marker="o", markersize=4)
    ax2.scatter([best_ep], [h["val_loss"][best_ep - 1]],
                color=color, s=130, zorder=5, edgecolors="white", linewidth=2,
                marker="*")

ax2.set_title("Autoencoder Validation Loss — Best Epoch Highlighted",
              fontsize=14, fontweight="bold", color=TITLE_COLOR, pad=12)
ax2.set_xlabel("Epoch", fontsize=12, color=LABEL_COLOR)
ax2.set_ylabel("Validation MSE Loss", fontsize=12, color=LABEL_COLOR)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.tick_params(colors=LABEL_COLOR)
ax2.grid(True, color=GRID_COLOR, alpha=0.5, linestyle="--")
ax2.legend(fontsize=10, loc="upper right", frameon=True,
           facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=LABEL_COLOR)

# ═══════════════════════════════════════════
# PANEL 3 — Metric Bars (Recall + Precision)
# ═══════════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(CARD_COLOR)

labels = [f"{int(p*100)}%" for p in DATA_PERCENTAGES]
x = np.arange(len(labels))
width = 0.18

for idx, model in enumerate(MODEL_NAMES):
    recalls = [all_results["model_metrics"][lbl][model]["recall"] for lbl in labels]
    offset = (idx - 1.5) * width
    bars = ax3.bar(x + offset, recalls, width, label=model,
                   color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                 f"{h:.3f}", ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color=LABEL_COLOR, rotation=90)

ax3.set_title("Recall by Training Percentage (on Fixed Test Set)",
              fontsize=14, fontweight="bold", color=TITLE_COLOR, pad=12)
ax3.set_xlabel("Training Data Percentage (of 70% Split)", fontsize=12, color=LABEL_COLOR)
ax3.set_ylabel("Recall", fontsize=12, color=LABEL_COLOR)
ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=11, fontweight="bold", color=LABEL_COLOR)
ax3.set_ylim(0, 1.15)
ax3.tick_params(colors=LABEL_COLOR)
ax3.grid(True, color=GRID_COLOR, alpha=0.5, linestyle="--", axis="y")
ax3.legend(fontsize=9, loc="upper left", frameon=True,
           facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=LABEL_COLOR,
           ncol=2)

# ═══════════════════════════════════════════
# PANEL 4 — F1-Score + PR-AUC Bars
# ═══════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(CARD_COLOR)

for idx, model in enumerate(MODEL_NAMES):
    f1s = [all_results["model_metrics"][lbl][model]["f1_score"] for lbl in labels]
    offset = (idx - 1.5) * width
    bars = ax4.bar(x + offset, f1s, width, label=model,
                   color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                 f"{h:.3f}", ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color=LABEL_COLOR, rotation=90)

ax4.set_title("F1-Score by Training Percentage (on Fixed Test Set)",
              fontsize=14, fontweight="bold", color=TITLE_COLOR, pad=12)
ax4.set_xlabel("Training Data Percentage (of 70% Split)", fontsize=12, color=LABEL_COLOR)
ax4.set_ylabel("F1-Score", fontsize=12, color=LABEL_COLOR)
ax4.set_xticks(x)
ax4.set_xticklabels(labels, fontsize=11, fontweight="bold", color=LABEL_COLOR)
ax4.set_ylim(0, 1.15)
ax4.tick_params(colors=LABEL_COLOR)
ax4.grid(True, color=GRID_COLOR, alpha=0.5, linestyle="--", axis="y")
ax4.legend(fontsize=9, loc="upper left", frameon=True,
           facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=LABEL_COLOR,
           ncol=2)

# ── Save ──
output_path = os.path.join(RESULTS_DIR, "v3_training_analysis_epochs.png")
fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
plt.close(fig)
print(f"\n✅ Training analysis chart saved: {output_path}")
print("═" * 70)
print("  TRAINING ANALYSIS COMPLETE")
print("═" * 70)
