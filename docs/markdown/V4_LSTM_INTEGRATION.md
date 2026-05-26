# V4 Hybrid Pipeline — LSTM Integration Guide

> **Precise and Enhancement of UPI Based Transaction Scam Identification using Hybrid Deep Learning Model**

This document explains every design decision, architecture choice, and implementation detail of the V4 experimental pipeline.

---

## Table of Contents

- [Why V4?](#why-v4)
- [What Changed from V3 to V4](#what-changed-from-v3-to-v4)
- [The Problem with V3's LSTM](#the-problem-with-v3s-lstm)
- [V4 Architecture](#v4-architecture)
- [BiLSTM with Bahdanau Attention](#bilstm-with-bahdanau-attention)
- [BiGRU with Attention (Comparison)](#bigru-with-attention-comparison)
- [SMOTE on Sequences](#smote-on-sequences)
- [20-Feature Ensemble](#20-feature-ensemble)
- [Enhanced Path B](#enhanced-path-b)
- [Protection Strategy](#protection-strategy)
- [File Map](#file-map)
- [How to Run](#how-to-run)
- [How to Evaluate](#how-to-evaluate)
- [Academic References](#academic-references)

---

## Why V4?

V3 achieves 99.6% recall and 100% precision — which is excellent. But it has a blind spot: **temporal patterns**. V3 treats each transaction independently. It cannot detect patterns like:

> "3 normal transactions → sudden large transfer → attempt to cash out immediately"

This sequential pattern is invisible to XGBoost and Random Forest, which see each transaction in isolation. LSTM (Long Short-Term Memory) networks are designed specifically to capture such temporal dependencies.

**V4's goal:** Add sequential pattern detection WITHOUT hurting V3's existing performance.

---

## What Changed from V3 to V4

```
V3 Pipeline (19 features):
  18 base features + AE reconstruction error → XGBoost + RF → Path A
  AE threshold + Isolation Forest → Path B

V4 Pipeline (20 features):
  18 base features + AE error + Sequential score → XGBoost + RF → Path A
  AE threshold + Isolation Forest + Sequential anomaly → Path B
                                   ↑ NEW: LSTM/GRU catches temporal fraud patterns
```

| Aspect | V3 | V4 |
|:-------|:---|:---|
| Features for XGBoost/RF | 19 (18 + AE error) | 20 (18 + AE error + seq_score) |
| Sequential model | None (LSTM was trained but unused) | BiLSTM or BiGRU with Attention |
| Path B signals | AE + IForest | AE + IForest + Sequential |
| LSTM training data | Unbalanced (0.13% fraud) | SMOTE-balanced (10% fraud) |
| LSTM features | 12 | 18 (same as XGBoost) |
| V3 models | — | **Completely untouched** |

---

## The Problem with V3's LSTM

V3's LSTM (trained in `src/train_lstm_model.py`) achieved only **48.9% recall** because of three issues:

### 1. Extreme Class Imbalance (No SMOTE)

```python
# V3 LSTM training — NO balancing
X = np.load("data/lstm_sequences/X.npy")  # 0.13% fraud
y = np.load("data/lstm_sequences/y.npy")  # Only focal loss for imbalance
```

With 0.13% fraud, the model learned to predict "legitimate" for everything and still achieved ~99.87% accuracy. Focal loss alone was insufficient.

**V4 fix:** SMOTE upsamples fraud sequences to 10% before training.

### 2. Limited Features (12 vs 18)

V3's LSTM used only 12 features, missing the 6 velocity features (`tx_count_cumul`, `amount_cumul`, `amt_vs_avg`, `time_since_last`, `amt_to_bal_ratio`, `balance_velocity`) — which are among the strongest fraud signals.

**V4 fix:** LSTM uses all 18 base features.

### 3. Basic Architecture

```python
# V3: Simple LSTM
LSTM(64) → Dropout(0.4) → Dense(32) → Dense(1)
```

No bidirectional processing, no attention mechanism, no batch normalization.

**V4 fix:** Bidirectional LSTM with Bahdanau Attention.

---

## V4 Architecture

### High-Level Flow

```
Transaction Data
      │
      ├──→ Feature Engineering (18 features)
      │            │
      │            ├──→ Autoencoder ──→ reconstruction error (19th signal)
      │            │
      │            ├──→ Sequential Model ──→ fraud score (20th signal)
      │            │         │
      │            │         └──→ Path B: sequential anomaly check
      │            │
      │            └──→ [18 + ae_error + seq_score] = 20 features
      │                          │
      │                 ┌────────┴────────┐
      │                 │                 │
      │            XGBoost(20)      Random Forest(20)
      │                 │                 │
      │                 └────┬────────────┘
      │                      │
      │              P = 0.5·XGB + 0.5·RF
      │                      │
      │              P ≥ threshold → PATH A: BLOCK
      │                      │
      │              Not blocked? → Check PATH B
      │                      │
      │              AE anomaly? OR IForest outlier? OR Sequential anomaly?
      │                      │
      │                Yes → PATH B: REVIEW
      │                      │
      │                No  → ALLOW
```

---

## BiLSTM with Bahdanau Attention

### Architecture Diagram

```
Input: (batch, 5 timesteps, 18 features)
        │
        ▼
┌──────────────────────────────────┐
│  Bidirectional LSTM (64 units)   │
│  ┌─── Forward LSTM(64) ──→ h_f  │
│  └─── Backward LSTM(64) ←── h_b │
│  Output: h = [h_f; h_b] per step│
│  Shape: (batch, 5, 128)         │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  Bahdanau Attention (32 units)   │
│  For each timestep t:            │
│    score_t = V · tanh(W · h_t)   │
│    α_t = softmax(score_t)        │
│  context = Σ(α_t · h_t)         │
│  Shape: (batch, 128)            │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  BatchNormalization              │
│  Dense(32, ReLU)                 │
│  Dropout(0.3)                    │
│  Dense(1, Sigmoid) → P(fraud)    │
└──────────────────────────────────┘
```

### Why Bidirectional?

Standard LSTM reads sequences left-to-right:
```
txn_1 → txn_2 → txn_3 → txn_4 → txn_5 → output
```

Bidirectional LSTM reads both directions:
```
Forward:  txn_1 → txn_2 → txn_3 → txn_4 → txn_5 → h_forward
Backward: txn_1 ← txn_2 ← txn_3 ← txn_4 ← txn_5 → h_backward
Output:   [h_forward; h_backward]  (concatenated)
```

This captures context from BOTH sides. If `txn_5` is suspicious, the backward pass can see "this is unusual given what comes BEFORE and AFTER it in the sequence."

### Why Attention?

Without attention, the LSTM compresses 5 timesteps into a single hidden state. Early timesteps may be "forgotten." Attention assigns a **learned weight** to each timestep:

```
Timestep weights (example):
  txn_1: 0.05  (low importance — normal transaction)
  txn_2: 0.08
  txn_3: 0.12
  txn_4: 0.25  (moderate — unusual amount)
  txn_5: 0.50  (highest — suspicious transfer)
```

The model learns WHICH timesteps are most fraud-indicative. This also provides **explainability** — we can inspect attention weights to understand WHY the model flagged a sequence.

### Bahdanau vs Luong Attention

We use **Bahdanau (Additive) attention**, not Luong (Multiplicative):

| Attention Type | Formula | When to Use |
|:---------------|:--------|:------------|
| **Bahdanau** | `V · tanh(W · h)` | Small sequences, lower dimensions |
| Luong | `h_t · W · h_s` | Longer sequences, higher dimensions |

Bahdanau is better for our case (5 timesteps, 128-dim hidden states) because it's more expressive for short sequences.

---

## BiGRU with Attention (Comparison)

GRU (Gated Recurrent Unit) is a simpler alternative to LSTM:

### LSTM vs GRU — Internal Mechanics

```
LSTM Cell:                          GRU Cell:
┌──────────────────────┐            ┌──────────────────────┐
│  Forget gate  f_t    │            │  Reset gate   r_t    │
│  Input gate   i_t    │            │  Update gate  z_t    │
│  Output gate  o_t    │            │  (no output gate)    │
│  Cell state   c_t    │            │  (no separate c_t)   │
│  Hidden state h_t    │            │  Hidden state h_t    │
│                      │            │                      │
│  Parameters: 4×(n²+n)│            │  Parameters: 3×(n²+n)│
└──────────────────────┘            └──────────────────────┘
```

**For our fraud detection task:**
- Sequences are SHORT (5 steps) — GRU's simplicity may generalize better
- GRU trains ~25% faster with ~25% fewer parameters
- On short sequences, GRU often matches or beats LSTM

The training script trains BOTH and picks the winner by PR-AUC on the validation set.

---

## SMOTE on Sequences

### The Problem

Raw PaySim has 0.13% fraud. After creating sequences:
- ~6 million sequences total
- ~8,000 fraud sequences (0.13%)
- The model sees ~770 legitimate sequences for every 1 fraud sequence

### The Solution

SMOTE (Synthetic Minority Oversampling Technique) creates synthetic fraud sequences:

```
Real fraud sequence A:     [txn_a1, txn_a2, txn_a3, txn_a4, txn_a5]
Real fraud sequence B:     [txn_b1, txn_b2, txn_b3, txn_b4, txn_b5]
                                        ↓ interpolate
Synthetic fraud sequence:  [0.6·a1+0.4·b1, 0.6·a2+0.4·b2, ..., 0.6·a5+0.4·b5]
```

**Implementation:**
1. Flatten sequences: `(samples, 5, 18) → (samples, 90)` — SMOTE needs 2D input
2. Apply SMOTE: creates synthetic fraud samples by interpolating between nearest-neighbor fraud sequences in the 90-dimensional space
3. Reshape back: `(samples_new, 90) → (samples_new, 5, 18)`
4. Result: ~10% fraud ratio (from 0.13%)

This preserves temporal structure because similar fraud sequences have similar temporal patterns $—$ interpolating between them creates plausible fraud sequences.

---

## 20-Feature Ensemble

### Feature Stack

| # | Feature | Source | Signal Type |
|:--|:--------|:-------|:------------|
| 1-18 | Base engineered features | Feature engineering | Individual transaction properties |
| 19 | `ae_recon_error` | Autoencoder | **Anomaly signal** — how unusual is this transaction? |
| 20 | `sequential_score` | BiLSTM/BiGRU | **Temporal signal** — how suspicious is the sequence? |

### Why Not Use LSTM as a Direct Voter?

Option A (rejected): `P = 0.33·XGB + 0.33·RF + 0.33·LSTM`
- LSTM's lower accuracy would drag down ensemble recall from 99.6% to ~80%
- Even with improvements, LSTM is unlikely to match XGBoost on this dataset

Option B (chosen): LSTM score as a FEATURE for XGBoost/RF
- XGBoost/RF learn HOW MUCH to trust the LSTM signal
- If LSTM is unreliable, XGBoost will assign it low feature importance
- If LSTM catches patterns others miss, XGBoost will leverage that signal
- **Zero risk of hurting Path A performance** — XGBoost decides the weight

---

## Enhanced Path B

V3 Path B: `flag if (AE_anomaly OR IForest_outlier) AND NOT auto_blocked`

V4 Path B: `flag if (AE_anomaly OR IForest_outlier OR Sequential_anomaly) AND NOT auto_blocked`

The sequential anomaly signal catches fraud patterns that AE and IForest miss:  
- **AE** catches unusual individual transactions (high reconstruction error)
- **IForest** catches statistical outliers in feature space
- **Sequential** catches suspicious temporal patterns (e.g., rapid succession of transfers)

---

## Protection Strategy

```
models/
├── paysim_v3/              ← LOCKED (never written to by V4 code)
│   ├── paysim_v3_xgb.pkl
│   ├── paysim_v3_rf.pkl
│   ├── paysim_v3_ae.keras
│   ├── paysim_v3_iforest.pkl
│   ├── paysim_v3_scaler.pkl
│   ├── paysim_v3_features.pkl
│   ├── paysim_v3_threshold.npy
│   ├── paysim_v3_ae_threshold.npy
│   └── paysim_v3_weights.npy
│
├── paysim_v4_experiment/   ← NEW (all V4 artifacts here)
│   ├── paysim_v4_xgb.pkl
│   ├── paysim_v4_rf.pkl
│   ├── paysim_v4_ae.keras
│   ├── paysim_v4_bilstm.keras
│   ├── paysim_v4_bigru.keras
│   ├── paysim_v4_sequential_winner.keras
│   ├── paysim_v4_iforest.pkl
│   ├── paysim_v4_base_scaler.pkl   (18 features)
│   ├── paysim_v4_scaler.pkl        (20 features)
│   ├── paysim_v4_features.pkl
│   ├── paysim_v4_features_20.pkl
│   ├── paysim_v4_threshold.npy
│   ├── paysim_v4_ae_threshold.npy
│   ├── paysim_v4_seq_threshold.npy
│   ├── paysim_v4_weights.npy
│   ├── paysim_v4_seq_length.pkl
│   ├── sequential_comparison.json
│   └── v4_training_results.json
```

**Safeguards:**
1. V4 training script ONLY writes to `models/paysim_v4_experiment/`
2. Backend defaults to V3 — V4 is opt-in via `model_version: "v4"` in API request
3. If V4 underperforms: `rm -rf models/paysim_v4_experiment/` — V3 is untouched
4. V4 model loader returns `None` if models don't exist — graceful fallback

---

## File Map

| File | Purpose | Status |
|:-----|:--------|:-------|
| `src/train_paysim_v4_hybrid.py` | V4 training pipeline (12 phases) | **NEW** |
| `src/v4_ensemble_inference.py` | V4 prediction (single + batch) | **NEW** |
| `src/model_loader.py` | Added `load_paysim_v4_hybrid()` | **MODIFIED** |
| `backend/schemas.py` | Added `model_version`, `seq_anomaly_score` | **MODIFIED** |
| `backend/app.py` | Added `seq_anomaly_score` to response | **MODIFIED** |
| `backend/inference.py` | V4 routing + lazy loading | **MODIFIED** |
| `experiments/evaluate_v4_vs_v3.py` | Head-to-head comparison script | **NEW** |

---

## How to Run

### Step 1: Train V4 Models

```bash
cd Fraud_Detection_Model_Paysim_CC

# Full training (recommended — takes ~15-30 min depending on hardware)
python src/train_paysim_v4_hybrid.py

# Quick test on 10% of data (5 min)
python src/train_paysim_v4_hybrid.py --data-pct 0.10

# Minimal smoke test (1 min)
python src/train_paysim_v4_hybrid.py --data-pct 0.05
```

### Step 2: Evaluate V4 vs V3

```bash
python experiments/evaluate_v4_vs_v3.py
```

### Step 3: Test API with V4

```bash
# Start backend
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8003 --reload

# Test V3 (default)
curl -X POST http://127.0.0.1:8003/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_type": "paysim", "tabular_features": [50000, 100000, 50000, 20000, 70000, 14, 3, 0, 0, 0, 0, 1, 0.69, 10.82, 1.0, 48, 0.41, -1.0]}'

# Test V4 (experimental)
curl -X POST http://127.0.0.1:8003/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_type": "paysim", "model_version": "v4", "tabular_features": [50000, 100000, 50000, 20000, 70000, 14, 3, 0, 0, 0, 0, 1, 0.69, 10.82, 1.0, 48, 0.41, -1.0]}'
```

---

## How to Evaluate

After training, check `models/paysim_v4_experiment/v4_training_results.json` for:

1. **Path A metrics**: Recall, Precision, F1 should be ≥ V3
2. **Sequential comparison**: `sequential_comparison.json` shows BiLSTM vs BiGRU winner
3. **Path B novel catch rate**: Should be > 90% (V3's baseline)

Run the comparison script for a complete head-to-head analysis.

---

## Academic References

| Concept | Paper | Year |
|:--------|:------|:-----|
| LSTM | Hochreiter & Schmidhuber — "Long Short-Term Memory" | 1997 |
| GRU | Cho et al. — "Learning Phrase Representations using RNN Encoder-Decoder" | 2014 |
| Bahdanau Attention | Bahdanau et al. — "Neural Machine Translation by Jointly Learning to Align and Translate" | 2015 |
| Bidirectional RNN | Schuster & Paliwal — "Bidirectional Recurrent Neural Networks" | 1997 |
| Focal Loss | Lin et al. — "Focal Loss for Dense Object Detection" | 2017 |
| SMOTE | Chawla et al. — "SMOTE: Synthetic Minority Over-sampling Technique" | 2002 |
| Isolation Forest | Liu et al. — "Isolation Forest" | 2008 |
| XGBoost | Chen & Guestrin — "XGBoost: A Scalable Tree Boosting System" | 2016 |

---

*Last updated: March 2026*
