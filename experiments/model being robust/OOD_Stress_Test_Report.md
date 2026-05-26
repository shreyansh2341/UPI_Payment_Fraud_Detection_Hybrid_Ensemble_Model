# Out-of-Distribution (OOD) Stress Test Report

## 1. Objective
To test if the model "memorized" a flaw in the synthetic data rather than learning fraud behavior. Specifically, the synthetic PaySim dataset frequently has `errorbalanceorig` ≠ 0 when fraud occurs. We forcefully neutralized these features to see if the model completely fails to identify fraud.

## 2. Methodology
1. We took all actual fraud transactions from the validation/test set.
2. We recorded the model's normal recall on these transactions.
3. We then artificially altered the dataset, setting `errorbalanceorig = 0.0` and `errorbalancedest = 0.0` for all fraud cases.
4. We passed these altered (OOD) cases back into the model to observe the drop in recall.

## 3. Results

| Test Condition | Recall on Test Frauds |
| :--- | :--- |
| **Natural Fraud Cases** | **98.10%** |
| **Neutralized Leaky Features (OOD)** | **96.98%** |

*Drop in detection rate: **1.12%***

## 4. Strict Interpretation
**The model is exceptionally robust against the data leak.** 
If the model had overfitted severely to the `errorbalance` anomaly, neutralizing it would have plummeted the recall closer to 0%. Instead, the recall only dropped by ~1.1%. 

This proves that the model's decision-making relies on the broader context of the transaction (e.g., `balance_velocity`, transaction amounts relative to average, etc.) and not just a single "cheat" variable.

## 5. Generalization Verdict
**Strong Generalization.** The stress test confirms that the XGBoost model has generalized the underlying *behavior* of the frauds (such as emptying an account rapidly) rather than simply memorizing the synthetic dataset's generation artifacts.
