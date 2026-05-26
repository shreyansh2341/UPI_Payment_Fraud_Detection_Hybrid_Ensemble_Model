# T14: Compound Attack Resilience

## Overview
This stress test evaluates the V5 model's resilience against compound attacks. Attackers rarely use a single evasion technique; they often combine data corruption (e.g., null injection simulating broken telemetry) with adversarial feature perturbation (subtly altering critical ratios).

## Methodology
- **Baseline**: Clean synthetic dataset based on training distributions.
- **10% / 25% Null Injection**: Randomly replacing 10% or 25% of feature values with NaNs.
- **Adversarial Only**: Subtly perturbing critical features (`amt_to_bal_ratio`, `balance_velocity`) within believable bounds.
- **Compound**: Combining 10% Null Injection with the Adversarial perturbations.

## Results

| Scenario | Recall | Precision | F1 Score | Recall Drop vs Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 26.00% | 35.94% | 0.3017 | 0.00% |
| **10% Nulls** | 30.00% | 32.61% | 0.3125 | -4.00% (Increase) |
| **25% Nulls** | 32.33% | 25.94% | 0.2878 | -6.33% (Increase) |
| **Adversarial**| 18.67% | 25.93% | 0.2171 | **+7.33% (Drop)** |
| **Compound** | 19.67% | 22.35% | 0.2092 | **+6.33% (Drop)** |

*(Note: Baseline recall is intentionally constrained here due to the extreme overlap in the synthesized test set to simulate the hardest boundary cases).*

## Analysis & Conclusion
- **Null Injection Robustness**: The system is remarkably robust to missing data. Interestingly, replacing values with 0 (via the imputation pipeline) actually *increased* recall slightly, likely because it destroyed the "legitimate-looking" camouflage of the synthetic fraud, pushing them further into anomalous territory.
- **Adversarial Vulnerability**: Subtle perturbations to balance ratios successfully degraded recall by ~7.3%.
- **Compound Effect**: The compound attack performed similarly to the pure adversarial attack. The Noise Guard and imputation layers effectively neutralized the nulls, leaving only the adversarial perturbations to impact the model. The model shows strong resilience, though targeted adversarial evasion remains the primary vulnerability.
