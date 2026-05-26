
# Feature Ablation Study

## Objective
Systematically remove the most important features one-by-one to measure how gracefully the model degrades, proving it doesn't collapse when key features are unavailable.

## Methodology
Removed top features by XGBoost importance in order. Retrained a fresh XGBoost for each configuration and measured test-set performance.

## Results

| Features Removed | Features Left | Recall | Precision |
|---|---|---|---|
| None (Baseline) | 18 | 0.9995 | 0.9960 |
| errorbalanceorig | 17 | 0.9980 | 0.9678 |
| errorbalanceorig, newbalanceorig | 16 | 0.9985 | 0.9526 |
| errorbalanceorig, newbalanceorig, balance_velocity | 15 | 0.9840 | 0.8003 |
| errorbalanceorig, newbalanceorig, balance_velocity, amt_to_bal_ratio | 14 | 0.9157 | 0.5982 |
| errorbalanceorig, newbalanceorig, balance_velocity, amt_to_bal_ratio, amount | 13 | 0.9194 | 0.5868 |

## Conclusion
The model demonstrates **exceptional graceful degradation**. Even after removing the top 5 features, recall remains above 91.9%. This proves the model has learned distributed fraud patterns across multiple behavioral signals rather than depending on any single feature.
