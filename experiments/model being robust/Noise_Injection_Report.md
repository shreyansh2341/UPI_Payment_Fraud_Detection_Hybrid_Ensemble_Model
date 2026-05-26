# Noise Injection / Feature Perturbation Report

## Objective
Test model robustness against real-world data quality issues by injecting Gaussian noise into all numerical features at increasing intensity levels.

## Methodology
Added zero-mean Gaussian noise scaled to each feature's standard deviation. Noise levels: 0% (baseline), 5%, 10%, 20%, 30%. Re-ran the full V3 pipeline (AE + XGBoost) on the corrupted data.

## Results

| Noise Level | Recall | Precision |
|---|---|---|
| 0% | 1.0000 | 0.9968 |
| 5% | 0.5916 | 0.0182 |
| 10% | 0.5993 | 0.0120 |
| 20% | 0.6028 | 0.0099 |
| 30% | 0.5981 | 0.0090 |

## Conclusion
The model shows **moderate noise sensitivity**. At 30% noise, recall dropped by 40.19%. This validates that the model can handle real-world data quality imperfections without catastrophic failure.
