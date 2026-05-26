# Stress Test 6: Adversarial Evasion (Smurfing Attack)

## Overview
Adversarial evasion involves actively trying to trick the AI. We simulated a specific type of attack called "Smurfing" or "Structuring." In this attack, a fraudster breaks down a large, highly suspicious transaction (e.g., ₹50,000) into many smaller, less suspicious transactions (e.g., 5 transactions of ₹10,000) to fly under the radar.

## Why We Perform It
To ensure the system can detect sophisticated, multi-step fraud campaigns, not just isolated, obvious attacks. Fraudsters constantly attempt to bypass threshold limits.

## How It Was Performed
We generated synthetic fraud sequences where the total stolen amount was divided evenly across a sequence length (e.g., 5 steps). We fed these sequences into the BiLSTM (our sequential detection model) to see if it could connect the dots and flag the overall sequence as fraudulent.

## Detailed Results
- **Attack Evasion Rate**: 46.00%
- **Evasions (False Negatives)**: 46 out of 100 simulated smurfing attacks successfully bypassed the BiLSTM layer.

## Conclusion
This test revealed an honest limitation in the BiLSTM component: highly uniform, mathematical splitting of amounts creates a synthetic pattern that the BiLSTM struggles to distinguish from automated payroll or subscription payments. **However, this weakness was the direct catalyst for the V5 Mitigations.** In the V5 pipeline, we implemented a `detect_smurfing_pattern()` Velocity Guard. Even if the BiLSTM is evaded, the rapid depletion of the account balance triggers an override, blocking the smurfing attack via the anomaly detection tier.
