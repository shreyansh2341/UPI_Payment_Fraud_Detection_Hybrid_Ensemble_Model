# Stress Test 2: Feature Importance and Ablation Study

## Overview
Feature Importance tells us how much each variable (like transaction amount, balance difference, etc.) contributes to the model's final decision. An Ablation Study involves systematically removing these features one by one to see how the model's performance degrades.

## Why We Perform It
To ensure our model isn't overly dependent on a single "magic bullet" feature. In real-world data, some features might be missing or corrupted. If a model relies 90% on one feature and that feature is unavailable, the model will fail entirely. We perform this to understand the model's decision-making process and guarantee it looks at a distributed set of signals.

## How It Was Performed
We used XGBoost's built-in feature importance metrics to rank features. We then systematically removed the top features (e.g., `errorbalanceorig`) and retrained/re-evaluated the model to observe the drop in recall and precision.

## Detailed Results
- `errorbalanceorig`: **43.02%**
- `newbalanceorig`: 18.93%
- `balance_velocity`: 15.54%
- `amt_to_bal_ratio`: 11.52%

## Conclusion
The report identified that `errorbalanceorig` is dominant at 43%. While this is a significant dependency, it is not a catastrophic failure (where one feature is >90%). Because this feature is a synthetic artifact of the PaySim dataset that might not exist in production, we validated the model's robustness by removing it. The model successfully fell back onto `balance_velocity` and `amt_to_bal_ratio`, maintaining high recall. This proves the feature set provides redundant layers of signal.
