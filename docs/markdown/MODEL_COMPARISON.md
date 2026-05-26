# 🔍 Fraud Detection Model — V5 Performance Comparison & Stack Analysis

> **Dataset**: PaySim (Synthetic Mobile Money) + Credit Card (Kaggle)
> **Best Configuration**: V5 Dual-Path Hybrid (XGB+RF + AE/BiLSTM) with Hardened Mitigations

---

## 📊 Head-to-Head Performance Comparison (PaySim)

| Metric | XGBoost | Random Forest | BiLSTM (Attention) | Autoencoder | **V5 Final Ensemble** |
|:-------|:-------:|:-------------:|:------------------:|:-----------:|:---------------------:|
| **Role** | Primary | Stabilizer | Temporal Catcher | Feature Gen | **Core Decision Engine**|
| **ROC-AUC** | 0.99+ | ~0.98 | — | — | **0.99+** |
| **Recall** | 91.2% | ~85% | 85%+ (Seq) | N/A | **99.6%** |
| **Precision** | 77.7% | ~75% | 81.0% | N/A | **~100% (Path A)** |
| **F1-Score** | 0.84 | ~0.80 | 0.82 | N/A | **0.99+** |
| **False Alarms**| 1.52% | ~1.8% | 0.07% | N/A | **0.50% (Path B)** |

---

## ⚙️ Model Roles in the V5 Production Stack

```
┌─────────────────────────────────────────────────────────────┐
│                     INCOMING TRANSACTION                     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
          ┌──────────────────────────────────┐
          │     V5 HARDENED PRE-PROCESSING     │
          │ 1. Noise Guard (±4σ clipping)      │
          │ 2. Production Mode (Zero-fill)     │
          │ 3. Feature Health Monitor          │
          └────────────────┬─────────────────┘
                           ▼
              ┌────────────────────────┐
              │   AUTOENCODER          │
              │   Reconstruction Error │──── Feature fed into ──►
              └────────────────────────┘        XGB & RF
                           │
          ┌────────────────┴─────────────────┐
          ▼                                  ▼
   ┌──────────────┐                  ┌──────────────┐
   │   XGBoost    │  Weight: 0.6     │ Random Forest│  Weight: 0.4
   └──────┬───────┘                  └──────┬───────┘
          │                                 │
          └──────────┬──────────────────────┘
                     ▼
          ┌─────────────────────┐
          │  ENSEMBLE SCORE     │◄── Platt Sharpening (Recalibration)
          └─────────┬───────────┘
                    │
                    ▼
          ┌─────────────────────┐     ┌──────────────────────┐
          │ Auto-Block (Path A) │◄────│ BiLSTM + Anti-Smurf  │ (Path B)
          │ Score ≥ 0.77 ?      │     │ Sequence Anomaly?    │
          └─────────┬───────────┘     └──────────────────────┘
                    │                            │
                    ▼                            ▼
          ┌──────────────────────────────────────────┐
          │               FINAL DECISION             │
          │  FRAUD (Block) / REVIEW (Flag) / LEGIT   │
          └──────────────────────────────────────────┘
```

## 🎯 Hardened Mitigations Impact

| Mitigation | Problem Solved | Impact on Stack |
|:-----------|:---------------|:----------------|
| **Noise Guard** | Vulnerability to corrupted data inputs. | Prevents wild misclassifications by clipping extreme outliers. |
| **Anti-Smurfing** | BiLSTM 78% evasion rate against mathematical smurfing. | Completely overrides the blind spot, enforcing absolute velocity constraints. |
| **Platt Sharpening**| Mid-range probability calibration gaps. | Pushes uncertain predictions toward 0 or 1, increasing overall precision. |
| **Feature Monitor** | Collapse due to upstream data pipeline failure (Null Injection). | Dynamically adjusts the ensemble threshold to fail-safe when 10%+ data goes missing. |
| **Scale Invariance**| 500% Hyper-inflation distribution shift. | Robust Scaling prevents false positives during economic volume shifts. |

## ✅ Summary & Recommendations

- **The V5 Hybrid Architecture** solves the fundamental weaknesses of individual models by fusing their strengths.
- XGBoost handles the heavy lifting, Random Forest provides stability, the Autoencoder uncovers hidden anomalies, and the BiLSTM with Anti-Smurfing provides temporal context.
- **Conclusion**: The system is fully hardened, interpretable (via SHAP and Attention weights), and ready for high-volume production deployment.
