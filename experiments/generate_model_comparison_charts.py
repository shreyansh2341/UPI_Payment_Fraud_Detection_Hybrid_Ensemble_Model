"""
generate_model_comparison_charts.py — Professional Metric Comparison Charts
═══════════════════════════════════════════════════════════════════════════════
Reads the training_analysis_results.json produced by generate_training_analysis.py
and generates 4 separate professional comparison charts:
  1. comparison_precision.png
  2. comparison_recall.png
  3. comparison_accuracy.png
  4. comparison_f1_score.png

Each chart compares ALL models (AE, RF, XGBoost, V3 Hybrid, LSTM) at the 70%
training percentage (best epoch configuration).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

RESULTS_DIR = "experiments/results"
RESULTS_JSON = os.path.join(RESULTS_DIR, "training_analysis_results.json")

# ══════════════════════════════════════════════════════════
# LSTM REFERENCE METRICS (FROM ARCHIVED TRAINING)
# ══════════════════════════════════════════════════════════
# These are the results from the LSTM model that was trained but excluded
# from the V3 hybrid pipeline due to underperformance
LSTM_METRICS = {
    "accuracy":  0.9975,  # weighted (dominated by legit)
    "precision": 0.810,
    "recall":    0.489,
    "f1_score":  0.610,
    "pr_auc":    0.5629,
}

# ══════════════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════════════
BG_COLOR     = "#0d1117"
CARD_COLOR   = "#161b22"
GRID_COLOR   = "#21262d"
TITLE_COLOR  = "#f0f6fc"
LABEL_COLOR  = "#c9d1d9"
BORDER_COLOR = "#30363d"

MODEL_COLORS = {
    "Autoencoder\n(Standalone)": "#f97583",
    "Random\nForest":            "#56d364",
    "XGBoost":                   "#d29922",
    "LSTM\n(Archived)":          "#bc8cff",
    "V3 Hybrid\nEnsemble":       "#58a6ff",
}

MODEL_ORDER = [
    "Autoencoder\n(Standalone)",
    "Random\nForest",
    "XGBoost",
    "LSTM\n(Archived)",
    "V3 Hybrid\nEnsemble",
]

METRIC_CONFIG = {
    "precision": {
        "title": "Precision Comparison — All Models",
        "ylabel": "Precision Score",
        "filename": "comparison_precision.png",
        "subtitle": "Higher is better • Measures how many flagged transactions are truly fraud",
        "color_accent": "#58a6ff",
    },
    "recall": {
        "title": "Recall Comparison — All Models",
        "ylabel": "Recall Score",
        "filename": "comparison_recall.png",
        "subtitle": "Higher is better • Measures how many real frauds are actually caught",
        "color_accent": "#56d364",
    },
    "accuracy": {
        "title": "Accuracy Comparison — All Models",
        "ylabel": "Accuracy Score",
        "filename": "comparison_accuracy.png",
        "subtitle": "Higher is better • Overall correctness across all transactions",
        "color_accent": "#d29922",
    },
    "f1_score": {
        "title": "F1-Score Comparison — All Models",
        "ylabel": "F1-Score",
        "filename": "comparison_f1_score.png",
        "subtitle": "Higher is better • Harmonic mean of Precision and Recall",
        "color_accent": "#f97583",
    },
}


def generate_metric_chart(metric_key, model_values, config, best_epoch_label, training_pct):
    """Generate a single professional metric comparison chart."""
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_COLOR)

    models = MODEL_ORDER
    values = [model_values[m] for m in models]
    colors = [MODEL_COLORS[m] for m in models]
    x = np.arange(len(models))

    # ── Bars with gradient effect ──
    bars = ax.bar(x, values, width=0.55, color=colors, edgecolor="white",
                  linewidth=1.2, zorder=3)

    # ── Value labels on top ──
    for i, (bar, val) in enumerate(zip(bars, values)):
        h = bar.get_height()
        pct_text = f"{val*100:.1f}%"
        raw_text = f"{val:.4f}"

        # Main percentage
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                pct_text, ha="center", va="bottom",
                fontsize=16, fontweight="bold", color=colors[i],
                path_effects=[pe.withStroke(linewidth=2, foreground=BG_COLOR)])

        # Raw score below percentage
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                raw_text, ha="center", va="top",
                fontsize=9, color=LABEL_COLOR, alpha=0.7)

    # ── Highlight best model ──
    best_idx = int(np.argmax(values))
    bars[best_idx].set_edgecolor("#ffd700")
    bars[best_idx].set_linewidth(3)

    ax.annotate("★ BEST", (x[best_idx], values[best_idx]),
                textcoords="offset points", xytext=(0, 35),
                ha="center", fontsize=11, fontweight="bold",
                color="#ffd700",
                arrowprops=dict(arrowstyle="->", color="#ffd700", lw=1.5))

    # ── Axis styling ──
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight="bold", color=LABEL_COLOR)
    ax.set_ylabel(config["ylabel"], fontsize=14, fontweight="bold",
                  color=LABEL_COLOR, labelpad=15)
    ax.set_ylim(0, max(values) + 0.12)
    ax.tick_params(axis="y", colors=LABEL_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.5, linestyle="--", axis="y", zorder=1)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)

    # ── Title ──
    fig.suptitle(config["title"], fontsize=20, fontweight="bold",
                 color=TITLE_COLOR, y=0.97)
    fig.text(0.5, 0.92, config["subtitle"],
             ha="center", fontsize=11, color=LABEL_COLOR, style="italic")

    # ── Footer info ──
    footer = (
        f"Dataset: PaySim (6.36M rows)  •  "
        f"Training: {training_pct} of 70% split  •  "
        f"Autoencoder best epoch: {best_epoch_label}  •  "
        f"Test set: 15% (fixed, untouched)"
    )
    fig.text(0.5, 0.02, footer, ha="center", fontsize=9, color=LABEL_COLOR, alpha=0.6)

    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.89])

    out_path = os.path.join(RESULTS_DIR, config["filename"])
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  ✅ {config['filename']}")
    return out_path


def main():
    print("═" * 70)
    print("  GENERATING MODEL PERFORMANCE COMPARISON CHARTS")
    print("═" * 70)

    # ── Load results ──
    if not os.path.exists(RESULTS_JSON):
        print(f"\n❌ ERROR: {RESULTS_JSON} not found!")
        print("   Run generate_training_analysis.py first.")
        return

    with open(RESULTS_JSON, "r") as f:
        results = json.load(f)

    # Use 70% training percentage as the reference (best config)
    ref_pct = "70%"
    if ref_pct not in results["model_metrics"]:
        # Fallback: use the highest available
        ref_pct = list(results["model_metrics"].keys())[-1]

    metrics_at_ref = results["model_metrics"][ref_pct]
    best_ae_epoch = results["best_ae_epoch"][ref_pct]

    print(f"\n  Reference training percentage: {ref_pct}")
    print(f"  Best AE epoch (at {ref_pct}): {best_ae_epoch}")

    # ── Build model values for each metric ──
    for metric_key, config in METRIC_CONFIG.items():
        model_values = {}
        for model_display, model_key in [
            ("Autoencoder\n(Standalone)", "Autoencoder (Standalone)"),
            ("Random\nForest",            "Random Forest"),
            ("XGBoost",                   "XGBoost"),
            ("LSTM\n(Archived)",          None),  # from LSTM_METRICS
            ("V3 Hybrid\nEnsemble",       "V3 Hybrid Ensemble"),
        ]:
            if model_key is None:
                model_values[model_display] = LSTM_METRICS[metric_key]
            else:
                model_values[model_display] = metrics_at_ref[model_key][metric_key]

        generate_metric_chart(
            metric_key, model_values, config,
            best_epoch_label=f"Epoch {best_ae_epoch}",
            training_pct=ref_pct,
        )

    print(f"\n{'═' * 70}")
    print(f"  ALL 4 COMPARISON CHARTS GENERATED SUCCESSFULLY!")
    print(f"{'═' * 70}")

    # Print summary table
    print(f"\n  Model Performance at {ref_pct} Training Data:")
    print(f"  {'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─' * 70}")
    for model_display, model_key in [
        ("AE (Standalone)",  "Autoencoder (Standalone)"),
        ("Random Forest",    "Random Forest"),
        ("XGBoost",          "XGBoost"),
        ("LSTM (Archived)",  None),
        ("V3 Hybrid",        "V3 Hybrid Ensemble"),
    ]:
        if model_key is None:
            m = LSTM_METRICS
        else:
            m = metrics_at_ref[model_key]
        print(f"  {model_display:<30} {m['accuracy']:>9.2%} {m['precision']:>9.2%} "
              f"{m['recall']:>9.2%} {m['f1_score']:>9.4f}")


if __name__ == "__main__":
    main()
