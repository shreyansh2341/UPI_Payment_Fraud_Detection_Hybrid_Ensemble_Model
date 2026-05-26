# Credit Card Fraud Detection — Model Comparison

> **Test Set**: 56,746 transactions (95 frauds, 0.167% fraud rate)

## Performance Metrics

| Metric | XGBoost | Random Forest | Ensemble (XGB+RF) |
|:-------|:-------:|:-------------:|:-----------------:|
| **Threshold** | 0.023529 | 0.463770 | 0.010200 |
| **ROC-AUC** | 0.9790 | 0.9740 | 0.9755 |
| **PR-AUC** | 0.8575 | 0.8324 | 0.8535 |
| **Accuracy** | 0.9933 | 0.9995 | 0.9534 |
| **Recall (Sensitivity)** | 0.8842 | 0.8421 | 0.9158 |
| **Precision** | 0.1846 | 0.8602 | 0.0319 |
| **F1-Score** | 0.3055 | 0.8511 | 0.0617 |
| **Specificity** | 0.9935 | 0.9998 | 0.9534 |
| **Alert Rate** | 0.8018% | 0.1639% | 4.8039% |

## Confusion Matrices

### XGBoost

```
                 Predicted
               Legit    Fraud
Actual Legit   56,280     371
Actual Fraud       11      84
```

### Random Forest

```
                 Predicted
               Legit    Fraud
Actual Legit   56,638      13
Actual Fraud       15      80
```

### Ensemble (XGB+RF)

```
                 Predicted
               Legit    Fraud
Actual Legit   54,012   2,639
Actual Fraud        8      87
```

---

*Evaluation run on credit card test set (56,746 samples).*
