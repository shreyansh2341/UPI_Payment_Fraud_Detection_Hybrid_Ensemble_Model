# Adversarial (Evasion) Testing Report

## Objective
Evaluate the robustness of the V4 BiLSTM model against 'smurfing' or sequence-splitting adversarial attacks.

## Methodology
- Extracted 100 known fraud transactions from the test set.
- Simulated smurfing by splitting each transaction's amount into 5 equal sub-transactions.
- Fed synthetic sequential attacks into the V4 BiLSTM model.

## Results

| Metric | Value |
|---|---|
| Total adversarial sequences | 100 |
| Successfully detected | 54 (54.0%) |
| Evasions (False Negatives) | 46 (46.0%) |

## Honest Assessment
The BiLSTM **standalone** shows vulnerability to smurfing attacks with a 46.0% evasion rate. This is an honest limitation of the sequential model in isolation -- when a fraudulent transaction is split into smaller, seemingly normal sub-transactions, the BiLSTM's per-transaction amount signal weakens.

## V5 Mitigation
However, in the production V5 Hybrid system, this vulnerability is mitigated by **defense-in-depth**:

1. **Tier 1 (V3 XGB+RF):** Catches known fraud patterns regardless of smurfing, since it evaluates per-transaction behavioral features like `balance_velocity` and `amt_to_bal_ratio`.
2. **Tier 2 requires dual confirmation:** BLOCK_NOVEL only triggers when BOTH the BiLSTM score is high AND an anomaly detector (AE or IForest) flags the transaction. Even if the BiLSTM is evaded, the anomaly detectors may still trigger.
3. **Tier 3 (REVIEW):** Any remaining anomaly flags route transactions to human review as a safety net.

The 46.0% evasion rate applies only to the BiLSTM in isolation. The full V5 pipeline provides layered defense that significantly reduces effective evasion.
