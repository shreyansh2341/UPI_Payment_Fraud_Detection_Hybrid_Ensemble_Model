# Data Cleaning & Preprocessing — V3 to V5 Pipeline

## 1. Introduction to Our Data Pipeline
Hello team! This document details our exact data cleaning and feature engineering pipeline. The quality of our V5 Hybrid Fraud Detection System relies entirely on the quality of the features we extract here. This guide explains not just *what* we did, but *why* we did it, serving as a blueprint for the entire feature engineering process.

## 2. Raw Dataset Context

We begin with the raw PaySim dataset (`PS_20174392719_1491204439457_log.csv`). It simulates mobile money transactions (similar to UPI/mobile wallets), containing **6,354,407 transactions** with **8,213 fraudulent cases** (0.13% fraud ratio).

### Raw Columns Overview
| Column | Type | Description |
|:-------|:-----|:------------|
| `step` | int | Time unit (1 step = 1 hour, 744 steps total) |
| `type` | str | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| `amount` | float | Transaction amount |
| `nameOrig` | str | Sender account ID |
| `oldbalanceOrg` | float | Sender balance before |
| `newbalanceOrig` | float | Sender balance after |
| `nameDest` | str | Receiver account ID |
| `oldbalanceDest` | float | Receiver balance before |
| `newbalanceDest` | float | Receiver balance after |
| `isFraud` | int | Ground truth label (0/1) |
| `isFlaggedFraud` | int | System flag (rarely set, mostly useless) |

---

## 3. The Cleaning & Engineering Steps

### Step 1: Column Standardization
- We lowercased all column names and stripped whitespace. 
- *Why?* Ensures consistent programmatic access across the entire ML pipeline, preventing frustrating key errors.

### Step 2: Transaction Type Filtering
- **Action**: Retained only **TRANSFER** and **CASH_OUT** types.
- *Why?* Exploratory Data Analysis (EDA) revealed that 100% of the fraud in this dataset occurs within these two transaction types. Removing the others drastically reduces dataset size and noise without losing a single fraudulent event.

### Step 3: UPI Type Encoding
- **Action**: Binary one-hot encoding into `upi_type_upi_payment` and `upi_type_upi_transfer`.
- *Why?* Machine learning models require numerical inputs. We map PaySim types to UPI equivalents (CASH_OUT → payment, TRANSFER → transfer) to make the model generalizable to real-world Indian/Global payment systems.

### Step 4: Temporal Feature Extraction
We extract human-interpretable time features from the raw `step` counter:
| Feature | Formula | Purpose |
|:--------|:--------|:--------|
| `hour` | `step % 24` | Captures time-of-day patterns (fraud often spikes at night). |
| `dayofweek` | `(step // 24) % 7` | Captures day-of-week trends. |
| `is_weekend` | `1 if dayofweek >= 5 else 0` | Fraudsters often target weekends when bank staff is lower. |

### Step 5: Balance Error Features (The "Smoking Gun")
| Feature | Formula |
|:--------|:--------|
| `errorbalanceorig` | `newbalanceorig + amount - oldbalanceorg` |
| `errorbalancedest` | `newbalancedest - amount - oldbalancedest` |

*Why?* In a legitimate transaction, the math must balance exactly. Fraudulent transactions often exhibit mathematical inconsistencies (e.g., draining an account completely but the balances don't mathematically align due to system overrides). These are the most predictive features in our dataset.

### Step 6: Velocity Features (Introduced in V3, critical for V4/V5)
These features track user behavior over time. They are crucial for the BiLSTM sequence model.
| Feature | Formula | Purpose |
|:--------|:--------|:--------|
| `tx_count_cumul` | `log1p(cumulative count per account)` | Transaction frequency (smurfing detection). |
| `amount_cumul` | `log1p(cumulative amount per account)` | Spending accumulation. |
| `amt_vs_avg` | `amount / running_average` | Deviation from normal spending behavior. |
| `time_since_last` | `step_diff per account (hours)` | Rapid succession detection. |
| `amt_to_bal_ratio` | `log1p(amount / balance)` | Captures account draining (fraudsters empty accounts). |
| `balance_velocity` | `(new_bal - old_bal) / amount` | Rate of balance change. |

### Step 7: Feature Removal (Preventing Data Leakage)
- `has_balance_mismatch` **REMOVED**: This feature was perfectly correlated with fraud due to an artifact in the simulator. Keeping it causes "data leakage" (the model cheats by finding the backdoor rather than learning fraud patterns).
- `nameOrig`, `nameDest` **REMOVED**: Account IDs are high-cardinality strings that cause overfitting and aren't useful for generalized prediction.
- `step`, `type`, `isFlaggedFraud` **REMOVED**: Replaced by our superior engineered temporal and categorical features.

---

## 4. Final Feature Set (18 Base Features)

After cleaning, every transaction is represented by these 18 numeric features:
```
amount, oldbalanceorg, newbalanceorig, oldbalancedest, newbalancedest,
hour, dayofweek, is_weekend, errorbalanceorig, errorbalancedest,
upi_type_upi_payment, upi_type_upi_transfer,
tx_count_cumul, amount_cumul, amt_vs_avg, time_since_last,
amt_to_bal_ratio, balance_velocity
```

> **Note on Feature #19**: The `ae_recon_error` is computed dynamically at inference time by our Autoencoder in Path B, bringing the total features fed into the XGBoost/RF ensemble to 19.

## 5. Output
The thoroughly cleaned and engineered data is saved as `cleaned_paysim_lstm.csv` (~763 MB). This file contains everything required for the V5 Supervised Ensemble and the Sequential BiLSTM pipelines.
