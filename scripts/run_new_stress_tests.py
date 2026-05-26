"""
run_new_stress_tests.py — Advanced Robustness Tests T13-T20 for Netra V5
==========================================================================
Self-contained: generates synthetic test data from scaler statistics.
Run from project root: python scripts/run_new_stress_tests.py
"""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_loader import load_paysim_hybrid, load_paysim_v4_hybrid
from src.robustness_mitigations import (
    noise_guard_sanitize,
    detect_smurfing_pattern,
    sharpen_probabilities,
    check_feature_health,
    zero_out_leaky_features,
    robust_clip_to_training_distribution,
)
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

OUT_DIR    = "docs/markdown/stress_tests"
RESULT_DIR = "results/new_stress_tests"
os.makedirs(OUT_DIR,    exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ── Load models ────────────────────────────────────────
print("Loading models...")
v3 = load_paysim_hybrid()
v4 = load_paysim_v4_hybrid()

V3_SCALER  = v3["scaler"]
V3_FEATS   = v3["features"]                     # 19 features incl. ae_recon_error
V3_BASE    = [f for f in V3_FEATS if f != "ae_recon_error"]   # 18 base features
THRESHOLD  = float(v3["block_threshold"])

# ── Build AE-only scaler (18 features) from V3 scaler ─
from sklearn.preprocessing import StandardScaler
ae_scaler = StandardScaler()
ae_scaler.mean_            = V3_SCALER.mean_[:len(V3_BASE)]
ae_scaler.scale_           = V3_SCALER.scale_[:len(V3_BASE)]
ae_scaler.var_             = V3_SCALER.var_[:len(V3_BASE)]
ae_scaler.n_features_in_   = len(V3_BASE)
ae_scaler.n_samples_seen_  = V3_SCALER.n_samples_seen_

# ── Synthetic dataset generator ─────────────────────────
RNG = np.random.default_rng(42)

def make_synthetic_dataset(n_legit=3000, n_fraud=300):
    """
    Generate synthetic transactions from scaler statistics (mean ± scale).
    Fraud samples have exaggerated balance errors and velocity anomalies.
    """
    rows = []

    def sample_row(is_fraud):
        r = {}
        for i, feat in enumerate(V3_BASE):
            mu    = float(V3_SCALER.mean_[i])
            sigma = float(V3_SCALER.scale_[i])
            val   = float(RNG.normal(mu, sigma * 0.5))
            r[feat] = val

        if is_fraud:
            # Strongly inflate fraud-specific features so the trained model detects them
            idx_err  = V3_BASE.index("errorbalanceorig")
            idx_vel  = V3_BASE.index("balance_velocity")
            idx_rat  = V3_BASE.index("amt_to_bal_ratio")
            mu_err   = float(V3_SCALER.mean_[idx_err])
            sig_err  = float(V3_SCALER.scale_[idx_err])
            mu_vel   = float(V3_SCALER.mean_[idx_vel])
            sig_vel  = float(V3_SCALER.scale_[idx_vel])
            mu_rat   = float(V3_SCALER.mean_[idx_rat])
            sig_rat  = float(V3_SCALER.scale_[idx_rat])
            # errorbalanceorig: extreme positive outlier (fraud drains source)
            r["errorbalanceorig"] = mu_err + sig_err * RNG.uniform(8, 15)
            # balance_velocity: extreme negative (balance plummets)
            r["balance_velocity"] = mu_vel - sig_vel * RNG.uniform(6, 10)
            # amt_to_bal_ratio: extreme positive (amount >> balance)
            r["amt_to_bal_ratio"] = mu_rat + sig_rat * RNG.uniform(6, 10)
        r["isfraud"] = int(is_fraud)
        return r

    for _ in range(n_legit):  rows.append(sample_row(False))
    for _ in range(n_fraud):  rows.append(sample_row(True))
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

print("Generating synthetic dataset...")
df_test = make_synthetic_dataset(n_legit=3000, n_fraud=300)
y_test  = df_test["isfraud"].values
X_test  = df_test[V3_BASE].values.astype(np.float64)
print(f"Synthetic test set: {len(df_test)} rows, {y_test.sum()} frauds")

# ── Core scoring helpers ───────────────────────────────
def score_v3(X_raw):
    """Return ensemble probabilities for a raw (unscaled) feature matrix."""
    X = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = ae_scaler.transform(X)
    Xs = robust_clip_to_training_distribution(Xs)
    rec = v3["ae"].predict(Xs, batch_size=2048, verbose=0)
    ae_err = np.log1p(np.mean(np.square(Xs - rec), axis=1))
    X19  = np.column_stack([X, ae_err])
    X19s = V3_SCALER.transform(X19)
    X19s = robust_clip_to_training_distribution(X19s)
    pxgb = v3["xgb"].predict_proba(X19s)[:, 1]
    prf  = v3["rf"].predict_proba(X19s)[:, 1]
    w    = v3["weights"]
    return w[0]*pxgb + w[1]*prf, ae_err

def metrics(y_true, probs, thr=None):
    thr = thr or THRESHOLD
    pred = (probs >= thr).astype(int)
    return (
        float(recall_score(y_true, pred, zero_division=0)),
        float(precision_score(y_true, pred, zero_division=0)),
        float(f1_score(y_true, pred, zero_division=0)),
    )

results = {}

# ═══════════════════════════════════════════════════════
# T13: Post-Mitigation Smurfing Re-Validation
# ═══════════════════════════════════════════════════════
print("\n[T13] Post-Mitigation Smurfing Re-Validation...")

def make_smurf_sequences(n_attacks=500, n_splits=5, base_amount=50000.0):
    rows = []
    for i in range(n_attacks):
        split_amt = base_amount / n_splits
        old_bal   = float(RNG.uniform(60000, 200000))
        for j in range(n_splits):
            new_bal = max(0.0, old_bal - split_amt)
            r = {f: 0.0 for f in V3_BASE}
            r["amount"]          = split_amt
            r["oldbalanceorg"]   = old_bal if "oldbalanceorg" in V3_BASE else 0.0
            r["newbalanceorig"]  = new_bal if "newbalanceorig" in V3_BASE else 0.0
            r["errorbalanceorig"] = 0.0
            r["errorbalancedest"] = 0.0
            r["balance_velocity"] = (new_bal - old_bal) / (split_amt + 1e-6)
            r["amt_to_bal_ratio"] = np.log1p(split_amt / (old_bal + 1e-6))
            r["tx_count_cumul"]   = np.log1p(j + 1)
            r["amount_cumul"]     = np.log1p(split_amt * (j + 1))
            r["amt_vs_avg"]       = 1.0
            r["time_since_last"]  = float(RNG.uniform(0.1, 1.5))
            r["attack_id"]        = i
            rows.append(r)
            old_bal = new_bal
    return pd.DataFrame(rows)

smurf_df = make_smurf_sequences(500)
smurf_X  = smurf_df[V3_BASE].fillna(0).values.astype(np.float64)

# BiLSTM sequential score (pre-mitigation)
v4_scaler  = v4["base_scaler"]
seq_length = v4["seq_length"]
seq_thr    = float(v4.get("seq_block_threshold", 0.5))

smurf_sc   = v4_scaler.transform(np.nan_to_num(smurf_X))
smurf_sc   = robust_clip_to_training_distribution(smurf_sc)
seqs       = np.array([np.tile(smurf_sc[i], (seq_length, 1)) for i in range(len(smurf_sc))], dtype=np.float32)
seq_scores = v4["sequential"].predict(seqs, batch_size=512, verbose=0).ravel()

# Anti-Smurfing guard
smurf_san  = noise_guard_sanitize(smurf_df[V3_BASE].fillna(0))
smurf_flag = detect_smurfing_pattern(smurf_san)

pre_evasion  = float((seq_scores < seq_thr).mean())
post_evasion = float(((seq_scores < seq_thr) & ~smurf_flag).mean())

results["T13"] = {
    "pre_mitigation_evasion_rate":  pre_evasion,
    "post_mitigation_evasion_rate": post_evasion,
    "smurfing_flag_rate":           float(smurf_flag.mean()),
    "improvement_pct":              (pre_evasion - post_evasion) * 100,
}
print(f"  Pre-mitigation evasion:  {pre_evasion:.2%}")
print(f"  Post-mitigation evasion: {post_evasion:.2%}")

# ═══════════════════════════════════════════════════════
# T14: Compound Attack (Null Injection + Adversarial)
# ═══════════════════════════════════════════════════════
print("\n[T14] Compound Attack...")

def inject_nulls(X, rate):
    Xc = X.copy().astype(float)
    mask = RNG.random(Xc.shape) < rate
    Xc[mask] = np.nan
    return Xc

def adversarial_perturb(X):
    Xc = X.copy().astype(float)
    idx_rat = V3_BASE.index("amt_to_bal_ratio")
    idx_vel = V3_BASE.index("balance_velocity")
    idx_err = V3_BASE.index("errorbalanceorig")
    Xc[:, idx_rat] *= RNG.uniform(0.05, 0.15, size=len(Xc))
    Xc[:, idx_vel] *= RNG.uniform(-0.1, 0.1, size=len(Xc))
    Xc[:, idx_err]  = 0.0
    return Xc

scenarios = {
    "Baseline":              X_test.copy(),
    "10pct_NullInjection":   inject_nulls(X_test, 0.10),
    "25pct_NullInjection":   inject_nulls(X_test, 0.25),
    "AdversarialOnly":       adversarial_perturb(X_test),
    "Compound_10pct+Adv":    adversarial_perturb(inject_nulls(X_test, 0.10)),
}

t14 = {}
for name, Xc in scenarios.items():
    df_t = pd.DataFrame(np.nan_to_num(Xc, nan=0.0), columns=V3_BASE)
    df_t = noise_guard_sanitize(df_t)
    probs, _ = score_v3(df_t.values.astype(np.float64))
    r, p, f  = metrics(y_test, probs)
    t14[name] = {"recall": r, "precision": p, "f1": f}
    print(f"  {name:30s}: Recall={r:.4f}  Precision={p:.4f}")

baseline_recall = t14["Baseline"]["recall"]
for k in t14:
    t14[k]["recall_drop_vs_baseline"] = round(baseline_recall - t14[k]["recall"], 4)

results["T14"] = t14

# ═══════════════════════════════════════════════════════
# T15: Calibration Verification (ECE)
# ═══════════════════════════════════════════════════════
print("\n[T15] Calibration Verification (ECE)...")

def compute_ece(y_true, probs, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0: continue
        acc  = y_true[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
    return float(ece)

probs_raw, _   = score_v3(X_test)
probs_sharp    = sharpen_probabilities(probs_raw)

ece_raw   = compute_ece(y_test, probs_raw)
ece_sharp = compute_ece(y_test, probs_sharp)

fop_r, mpv_r = calibration_curve(y_test, probs_raw,   n_bins=10, strategy="uniform")
fop_s, mpv_s = calibration_curve(y_test, probs_sharp, n_bins=10, strategy="uniform")

results["T15"] = {
    "ece_raw":         ece_raw,
    "ece_sharpened":   ece_sharp,
    "ece_improvement": (ece_raw - ece_sharp) * 100,
    "calibration_bins": {
        "raw":      {"fraction_positives": fop_r.tolist(), "mean_predicted": mpv_r.tolist()},
        "sharpened":{"fraction_positives": fop_s.tolist(), "mean_predicted": mpv_s.tolist()},
    }
}
print(f"  ECE raw:       {ece_raw:.4f}")
print(f"  ECE sharpened: {ece_sharp:.4f}  (delta={results['T15']['ece_improvement']:+.2f}%)")

# ═══════════════════════════════════════════════════════
# T16: Synthetic Concept Drift
# ═══════════════════════════════════════════════════════
print("\n[T16] Synthetic Concept Drift...")

# 5 temporal bins; drift injected in bins 4-5
n_bins   = 5
bin_size = len(df_test) // n_bins
t16      = {}

for b in range(n_bins):
    start = b * bin_size
    end   = (b+1) * bin_size if b < n_bins-1 else len(df_test)
    Xb    = X_test[start:end].copy()
    yb    = y_test[start:end]

    if b >= 3 and yb.sum() > 0:
        fraud_idx = np.where(yb == 1)[0]
        idx_rat   = V3_BASE.index("amt_to_bal_ratio")
        idx_vel   = V3_BASE.index("balance_velocity")
        idx_err   = V3_BASE.index("errorbalanceorig")
        Xb[np.ix_(fraud_idx, [idx_rat])] *= 0.05
        Xb[np.ix_(fraud_idx, [idx_vel])] *= -0.05
        Xb[np.ix_(fraud_idx, [idx_err])]  = 0.0

    df_b = noise_guard_sanitize(pd.DataFrame(Xb, columns=V3_BASE))
    probs_b, _ = score_v3(df_b.values.astype(np.float64))
    r, p, _ = metrics(yb, probs_b)

    t16[f"Bin_{b+1}"] = {
        "n_samples": int(end-start),
        "n_fraud":   int(yb.sum()),
        "recall":    r,
        "precision": p,
        "drift_injected": b >= 3,
    }
    print(f"  Bin {b+1}: Recall={r:.4f}  Precision={p:.4f}  Drift={b>=3}")

results["T16"] = t16

# ═══════════════════════════════════════════════════════
# T17: Noise Guard Edge Case Audit
# ═══════════════════════════════════════════════════════
print("\n[T17] Noise Guard Edge Case Audit...")

# Legitimate high-value outliers (amount > 3σ from mean)
legit_idx  = np.where(y_test == 0)[0]
amt_col    = V3_BASE.index("amount")
amt_vals   = X_test[legit_idx, amt_col]
amt_thr    = amt_vals.mean() + 3 * amt_vals.std()
hv_idx     = legit_idx[amt_vals > amt_thr]

if len(hv_idx) < 5:
    hv_idx = legit_idx[np.argsort(amt_vals)[-200:]]

X_hv       = X_test[hv_idx]
y_hv       = y_test[hv_idx]
probs_hv, _= score_v3(X_hv)
fpr_hv     = float((probs_hv >= THRESHOLD)[y_hv == 0].mean())

# Adversarial noise injection (>6σ spike then clipped)
X_noisy    = X_test.copy().astype(float)
noise_mask = RNG.random(X_noisy.shape) < 0.05
X_noisy[noise_mask] = RNG.uniform(1e7, 1e8, size=noise_mask.sum())
max_pre    = float(np.abs(X_noisy).max())
df_clipped = noise_guard_sanitize(pd.DataFrame(X_noisy, columns=V3_BASE))
max_post   = float(np.abs(df_clipped.values).max())

# Verify clipped data still detects fraud correctly
probs_n, _ = score_v3(df_clipped.values.astype(np.float64))
r_n, p_n, _ = metrics(y_test, probs_n)

results["T17"] = {
    "n_high_value_legit":        int(len(hv_idx)),
    "fpr_high_value_legit":      fpr_hv,
    "adversarial_max_pre_clip":  max_pre,
    "adversarial_max_post_clip": max_post,
    "clip_reduction_factor":     max_pre / (max_post + 1e-6),
    "recall_after_noise_guard":  r_n,
    "precision_after_noise_guard": p_n,
}
print(f"  FPR on high-value legit txns: {fpr_hv:.4f}")
print(f"  Noise clipped: {max_pre:.2e} -> {max_post:.2e}")
print(f"  Post-clip recall: {r_n:.4f}")

# ═══════════════════════════════════════════════════════
# T18: Behavioral Profile Fairness
# ═══════════════════════════════════════════════════════
print("\n[T18] Behavioral Profile Fairness...")

# Reconstruct approximate tx_count from feature if available
if "tx_count_cumul" in V3_BASE:
    tx_idx     = V3_BASE.index("tx_count_cumul")
    raw_count  = np.expm1(X_test[:, tx_idx])
else:
    raw_count  = RNG.integers(1, 100, size=len(y_test)).astype(float)

if "time_since_last" in V3_BASE:
    tsl_idx   = V3_BASE.index("time_since_last")
    time_gap  = X_test[:, tsl_idx]
else:
    time_gap  = RNG.uniform(0, 100, size=len(y_test))

masks = {
    "New_Account_lt5_txns":   raw_count < 5,
    "Active_5to50_txns":      (raw_count >= 5) & (raw_count < 50),
    "Power_User_gt50_txns":   raw_count >= 50,
    "Dormant_gap_gt72h":      time_gap > 72,
    "Active_gap_lt2h":        time_gap < 2,
}

t18 = {}
for pname, mask in masks.items():
    if mask.sum() < 10: continue
    Xp  = X_test[mask]
    yp  = y_test[mask]
    if yp.sum() == 0 or (1-yp).sum() == 0: continue
    probs_p, _ = score_v3(Xp)
    pred_p     = (probs_p >= THRESHOLD).astype(int)
    fpr_p      = float(pred_p[yp==0].mean())
    rec_p      = float(pred_p[yp==1].mean())
    t18[pname] = {"n": int(mask.sum()), "n_fraud": int(yp.sum()),
                   "false_positive_rate": fpr_p, "recall": rec_p}
    print(f"  {pname:30s}: FPR={fpr_p:.4f}  Recall={rec_p:.4f}  (n={mask.sum()})")

results["T18"] = t18

# Fairness disparity metric
if t18:
    fprs = [v["false_positive_rate"] for v in t18.values()]
    recs = [v["recall"]              for v in t18.values()]
    results["T18"]["max_fpr_disparity"]    = float(max(fprs) - min(fprs))
    results["T18"]["max_recall_disparity"] = float(max(recs) - min(recs))

# ═══════════════════════════════════════════════════════
# T19: MC Dropout Uncertainty Estimation
# ═══════════════════════════════════════════════════════
print("\n[T19] MC Dropout Uncertainty Estimation...")

N_MC   = 30
N_SAMP = min(1500, len(df_test))
idx_s  = RNG.choice(len(df_test), N_SAMP, replace=False)
Xs_mc  = X_test[idx_s]
ys_mc  = y_test[idx_s]

Xs_sc  = v4_scaler.transform(np.nan_to_num(Xs_mc, nan=0.0))
Xs_sc  = robust_clip_to_training_distribution(Xs_sc)
seqs_mc = np.array([np.tile(Xs_sc[i], (seq_length, 1)) for i in range(N_SAMP)], dtype=np.float32)

mc_preds = np.zeros((N_MC, N_SAMP))
for k in range(N_MC):
    mc_preds[k] = v4["sequential"](seqs_mc, training=True).numpy().ravel()

mc_mean = mc_preds.mean(axis=0)
mc_var  = mc_preds.var(axis=0)
var_thr = float(np.percentile(mc_var, 95))

# Adversarial samples
Xadv_mc  = adversarial_perturb(Xs_mc)
Xadv_sc  = v4_scaler.transform(np.nan_to_num(Xadv_mc, nan=0.0))
Xadv_sc  = robust_clip_to_training_distribution(Xadv_sc)
seqs_adv = np.array([np.tile(Xadv_sc[i], (seq_length, 1)) for i in range(N_SAMP)], dtype=np.float32)
mc_adv   = np.zeros((N_MC, N_SAMP))
for k in range(N_MC):
    mc_adv[k] = v4["sequential"](seqs_adv, training=True).numpy().ravel()
mc_var_adv = mc_adv.var(axis=0)

results["T19"] = {
    "n_samples":                       N_SAMP,
    "n_mc_passes":                     N_MC,
    "mean_variance_legit":             float(mc_var[ys_mc==0].mean()),
    "mean_variance_fraud":             float(mc_var[ys_mc==1].mean()) if ys_mc.sum()>0 else None,
    "mean_variance_adversarial":       float(mc_var_adv.mean()),
    "variance_95th_threshold":         var_thr,
    "pct_adversarial_high_uncertainty": float((mc_var_adv > var_thr).mean()),
    "pct_legit_high_uncertainty":       float((mc_var[ys_mc==0] > var_thr).mean()),
}
print(f"  Mean Var (legit):       {results['T19']['mean_variance_legit']:.6f}")
print(f"  Mean Var (adversarial): {results['T19']['mean_variance_adversarial']:.6f}")

# ═══════════════════════════════════════════════════════
# T20: Latency & Throughput Benchmarking
# ═══════════════════════════════════════════════════════
print("\n[T20] Latency & Throughput Benchmarking...")

N_BENCH  = 1000
X_bench  = X_test[:N_BENCH]

# Single-transaction latency (200 warmup runs)
lat_times = []
for i in range(200):
    Xr = X_bench[i:i+1]
    t0 = time.perf_counter()
    df_s = noise_guard_sanitize(pd.DataFrame(Xr, columns=V3_BASE))
    _    = score_v3(df_s.values.astype(np.float64))
    lat_times.append(time.perf_counter() - t0)

lat_ms = np.array(lat_times) * 1000

# Batch throughput
t0_batch = time.perf_counter()
df_batch = noise_guard_sanitize(pd.DataFrame(X_bench, columns=V3_BASE))
_        = score_v3(df_batch.values.astype(np.float64))
_        = detect_smurfing_pattern(df_batch)
elapsed  = time.perf_counter() - t0_batch
throughput = N_BENCH / elapsed

results["T20"] = {
    "latency_p50_ms":          float(np.percentile(lat_ms, 50)),
    "latency_p95_ms":          float(np.percentile(lat_ms, 95)),
    "latency_p99_ms":          float(np.percentile(lat_ms, 99)),
    "latency_mean_ms":         float(lat_ms.mean()),
    "batch_1000_elapsed_s":    float(elapsed),
    "throughput_txns_per_sec": float(throughput),
    "meets_50ms_p95_target":   bool(np.percentile(lat_ms, 95) < 50),
    "meets_10k_tps_target":    bool(throughput > 10000),
}
print(f"  P50={results['T20']['latency_p50_ms']:.1f}ms  "
      f"P95={results['T20']['latency_p95_ms']:.1f}ms  "
      f"Throughput={throughput:.0f} txns/s")

# ── Save raw JSON ──────────────────────────────────────
out_path = os.path.join(RESULT_DIR, "new_stress_test_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nAll results saved to: {out_path}")
print("Run write_new_stress_reports.py next to generate the markdown reports.")
