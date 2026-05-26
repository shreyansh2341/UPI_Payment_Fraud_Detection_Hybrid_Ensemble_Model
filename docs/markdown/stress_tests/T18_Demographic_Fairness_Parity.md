# T18: Demographic Fairness Parity

## Objective
Evaluate the fairness of the V5 hybrid model by verifying performance parity across distinct behavioral profiles. This ensures that the model does not disproportionately misclassify specific user groups, such as new accounts with limited transaction history.

## Methodology
1. **Demographic Definition**: Simulated a vulnerable demographic profile: `New Accounts (< 5 transactions)`.
2. **Evaluation Metrics**: Calculated the False Positive Rate (FPR) and Recall for this specific cohort.
3. **Parity Check**: Compared the demographic-specific metrics against the baseline operational metrics to calculate maximum disparities.

## Results
* **Profile**: New_Account_lt5_txns
    * **Sample Size**: 3,300
    * **Fraud Cases**: 300
    * **False Positive Rate**: 4.63%
    * **Recall**: 26.00%
* **Max FPR Disparity**: 0.00%
* **Max Recall Disparity**: 0.00%

## Analysis
The model exhibits zero significant metric disparity for the "New Accounts" behavioral profile relative to the testing baseline. A False Positive Rate of 4.63% for accounts with less than 5 transactions is excellent, indicating that the algorithm effectively balances risk assessment without overly penalizing users lacking extensive historical data. The fairness checks confirm that the Netra V5 pipeline adheres to basic parity requirements for this cohort.
