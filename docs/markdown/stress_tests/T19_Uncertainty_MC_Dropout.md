# T19: Uncertainty Estimation (MC Dropout)

## Objective
Measure the model's confidence and predictive variance during adversarial scenarios using Monte Carlo (MC) Dropout. This provides a quantifiable measure of epistemic uncertainty, allowing the system to flag transactions where the model is fundamentally unsure.

## Methodology
1. **Test Size**: Evaluated on a sample of 1,500 transactions combining legitimate, fraudulent, and adversarially perturbed data.
2. **MC Passes**: Conducted 30 stochastic forward passes per transaction with dropout enabled.
3. **Variance Analysis**: Calculated the predictive variance across the passes for each class (Legit, Fraud, Adversarial).
4. **Uncertainty Flagging**: Established a 95th-percentile variance threshold and measured what percentage of transactions exceeded this boundary.

## Results
* **Total Samples Evaluated**: 1,500
* **MC Forward Passes**: 30
* **Mean Predictive Variance (Legitimate)**: 0.00386
* **Mean Predictive Variance (Fraudulent)**: 0.00392
* **Mean Predictive Variance (Adversarial)**: 0.00377
* **95th Percentile Uncertainty Threshold**: 0.01685
* **% of Adversarial Txns with High Uncertainty**: 5.13%
* **% of Legitimate Txns with High Uncertainty**: 4.78%

## Analysis
The Monte Carlo Dropout analysis indicates relatively low mean predictive variance across all transaction classes (~0.0038). Interestingly, adversarial examples did not yield significantly higher mean variance than standard fraudulent or legitimate transactions in this test iteration.
Approximately 5.13% of adversarial transactions and 4.78% of legitimate transactions breached the 95th-percentile uncertainty threshold (0.01685). This suggests that while MC Dropout can identify a small subset of highly ambiguous transactions for manual review, the current V5 architecture is highly confident (low variance) in its predictions across the board, even when facing perturbed inputs.
