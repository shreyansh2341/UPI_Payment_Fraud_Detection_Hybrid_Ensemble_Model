# Model Stack — V3 Hybrid

## PaySim Fraud Detection (Active: `models/paysim_v3/`)

### Features (19 total)

| # | Feature | Description |
|:--|:--------|:------------|
| 1 | `amount` | Transaction amount |
| 2 | `oldbalanceorg` | Sender's balance before |
| 3 | `newbalanceorig` | Sender's balance after |
| 4 | `oldbalancedest` | Receiver's balance before |
| 5 | `newbalancedest` | Receiver's balance after |
| 6 | `hour` | Hour of transaction (0-23) |
| 7 | `dayofweek` | Day of week (0-6) |
| 8 | `is_weekend` | Weekend flag (0/1) |
| 9 | `errorbalanceorig` | Balance error (sender) |
| 10 | `errorbalancedest` | Balance error (receiver) |
| 11 | `upi_type_upi_payment` | UPI payment flag |
| 12 | `upi_type_upi_transfer` | UPI transfer flag |
| 13 | `tx_count_cumul` | log1p(cumulative tx count) |
| 14 | `amount_cumul` | log1p(cumulative amount) |
| 15 | `amt_vs_avg` | Amount vs running average |
| 16 | `time_since_last` | Hours since last transaction |
| 17 | `amt_to_bal_ratio` | log1p(amount / balance) |
| 18 | `balance_velocity` | Balance change rate |
| 19 | `ae_recon_error` | Autoencoder reconstruction error (computed at inference) |

### Models

| Model | Type | Role |
|:------|:-----|:-----|
| `paysim_v3_xgb.pkl` | XGBoost | Path A: supervised classifier (19 features) |
| `paysim_v3_rf.pkl` | Random Forest | Path A: supervised classifier (19 features) |
| `paysim_v3_ae.keras` | Autoencoder (64→32→8→32→64) | Computes feature #19 + Path B anomaly |
| `paysim_v3_iforest.pkl` | Isolation Forest | Path B: outlier detection |

### Decision Logic

```
Score = 0.5 × XGB_prob + 0.5 × RF_prob

if Score >= 0.77:        → BLOCK (auto-block)
elif AE_err >= 0.052 or IForest == outlier:  → REVIEW (flag for analyst)
else:                    → ALLOW
```

### Performance

| Metric | Value |
|:-------|:------|
| Known fraud recall | 99.6% |
| Known fraud precision | 100.0% |
| Novel fraud flagged | 90% |
| False block rate | 0 |
| Review false positive rate | 0.50% |

---

## Credit Card Detection (Active: `models/creditcard/`)

| Model | Type |
|:------|:-----|
| `cc_xgb_model.pkl` | XGBoost |
| `cc_rf_model.pkl` | Random Forest |

Weighted ensemble with optimized threshold.

---

## Archived Models (`models/archive/`)

| Version | Contents | Notes |
|:--------|:---------|:------|
| V1 (Stage 2) | XGB, RF, AE, LSTM | Original 12-feature pipeline |
| V2 | XGB, RF, AE, IForest, LSTM | Retrained on full 6.36M dataset |
| V2 Baseline | XGB, RF | Backup copy of v2 |

All archived models are preserved for experiment documentation.
