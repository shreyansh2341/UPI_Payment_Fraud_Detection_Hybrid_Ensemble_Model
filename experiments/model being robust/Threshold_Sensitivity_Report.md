# Threshold Sensitivity & PR Calibration Report

## Objective
Analyze the Precision-Recall tradeoff across different operating thresholds to facilitate ROI-based business decisions (e.g., minimizing friction vs. maximizing fraud capture).

## V3 XGBoost Threshold Calibration
| Threshold | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| 0.10 | 0.9662 | 1.0000 | 0.9828 |
| 0.30 | 0.9933 | 1.0000 | 0.9966 |
| 0.50 | 0.9968 | 1.0000 | 0.9984 |
| 0.70 | 0.9990 | 1.0000 | 0.9995 |
| 0.90 | 1.0000 | 1.0000 | 1.0000 |
| 0.95 | 1.0000 | 1.0000 | 1.0000 |

## Conclusion
The model exhibits a sharp precision-recall curve. Lowering the threshold aggressively captures almost all frauds but incurs a precision cost (more manual reviews). A higher threshold (0.9+) guarantees ultra-high precision, making it ideal for the auto-block tier. This confirms the multi-tier strategy is optimal.
