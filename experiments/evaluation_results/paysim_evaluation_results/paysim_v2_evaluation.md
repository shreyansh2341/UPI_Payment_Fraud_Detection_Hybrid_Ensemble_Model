# PaySim v2 — Model Evaluation Results

> **Trained on**: Full cleaned_paysim.csv (4,453,674 rows)
> **SMOTE**: Fraud boosted from 0.129% to 9.09% (training only)
> **Threshold**: 0.6000 (derived via F2-score)
> **Weights**: 0.5 XGB + 0.5 RF

## Performance Metrics (Test Set)

| Metric | XGBoost | Random Forest | Ensemble |
|:-------|:-------:|:-------------:|:--------:|
| **ROC-AUC** | 0.9997 | 0.9997 | 0.9997 |
| **PR-AUC** | 0.9887 | 0.9984 | 0.9979 |
| **Recall** | 0.9911 | 0.9976 | 0.9951 |
| **Precision** | 0.8151 | 1.0000 | 0.9959 |
| **F1** | 0.8945 | 0.9988 | 0.9955 |
| **F2** | 0.9500 | 0.9981 | 0.9953 |
| **Accuracy** | 0.9997 | 1.0000 | 1.0000 |
| **Specificity** | 0.9997 | 1.0000 | 1.0000 |

## Confusion Matrices (Test Set)

### XGBoost

```
                 Predicted
               Legit    Fraud
Actual Legit    952,884     277
Actual Fraud         11   1,221
```

### Random Forest

```
                 Predicted
               Legit    Fraud
Actual Legit    953,161       0
Actual Fraud          3   1,229
```

### Ensemble

```
                 Predicted
               Legit    Fraud
Actual Legit    953,156       5
Actual Fraud          6   1,226
```

