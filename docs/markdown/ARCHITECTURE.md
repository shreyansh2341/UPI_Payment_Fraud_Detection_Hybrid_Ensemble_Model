# System Architecture — V3 Hybrid

## Overview

The system detects UPI/PaySim fraud using a **hybrid two-path architecture**:

- **Path A (Auto-Block)**: High-precision supervised models auto-block confirmed fraud
- **Path B (Flag for Review)**: Anomaly detectors flag suspicious transactions for human analysts

Credit card fraud detection uses a separate XGBoost+RF ensemble.

## Architecture Diagram

```
                        ┌────────────────────────────┐
                        │    Transaction Input        │
                        │  (API / CSV Upload / UI)    │
                        └─────────────┬──────────────┘
                                      │
                              ┌───────▼───────┐
                              │ Format Detect  │
                              │ (inference.py) │
                              └───┬───────┬───┘
                                  │       │
                     PaySim       │       │    CreditCard
                  ┌───────────────┘       └────────────────┐
                  ▼                                        ▼
        ┌─────────────────┐                     ┌─────────────────┐
        │ Feature Engineer │                     │ CC XGB+RF       │
        │ (18 base + AE)   │                     │ Ensemble        │
        └────────┬────────┘                     └────────┬────────┘
                 │                                       │
      ┌──────────┼──────────┐                   FRAUD / LEGIT
      │          │          │
      ▼          ▼          ▼
  ┌────────┐ ┌────────┐ ┌────────────┐
  │ PATH A │ │ PATH B │ │  AE Error  │
  │ XGB+RF │ │ AE +   │ │ (Feature   │
  │(19 ft) │ │ IForest│ │   #19)     │
  └───┬────┘ └───┬────┘ └────────────┘
      │          │
      ▼          ▼
  ┌────────┐ ┌────────┐
  │ BLOCK  │ │ REVIEW │
  │(≥0.77) │ │(Anomaly│
  │        │ │  Flag) │
  └────────┘ └────────┘
```

## Components

### FastAPI Backend (`backend/`)
| File | Purpose |
|:-----|:--------|
| `app.py` | FastAPI application, `/predict` endpoint |
| `inference.py` | Format detection, feature engineering, routing |
| `schemas.py` | Request/Response Pydantic models |

### Core ML (`src/`)
| File | Purpose |
|:-----|:--------|
| `model_loader.py` | Loads models from `models/paysim_v3/` and `models/creditcard/` |
| `final_ensemble_inference.py` | Hybrid prediction logic (Path A + Path B) |
| `utils/preprocessor.py` | Raw PaySim preprocessing |

### Models (`models/`)
| Directory | Contents |
|:----------|:---------|
| `paysim_v3/` | **Active** — XGB, RF, AE, IForest, scaler, thresholds (19 features) |
| `creditcard/` | CC XGB + RF ensemble |
| `archive/` | V1 and V2 models (preserved for reference) |

### Hybrid Detector (`hybrid_fraud_detector.py`)
Standalone class for batch scoring. Used by evaluation scripts.

## API Response Format

```json
{
  "decision": "BLOCK | REVIEW | ALLOW",
  "explanation": "Ensemble=0.92 (XGB=0.95, RF=0.89) | AE=0.03",
  "confidence": 0.92,
  "review_flag": false,
  "ae_anomaly_score": 0.03
}
```

## Performance (V3 Hybrid)

| Metric | Known Fraud | Novel Fraud |
|:-------|:-----------:|:-----------:|
| Recall | 99.6% (auto-blocked) | 90% (flagged for review) |
| Precision | 100% | N/A (human reviews) |
| False blocks | 0 | 0 |
| Review FP rate | — | 0.50% of legit |
