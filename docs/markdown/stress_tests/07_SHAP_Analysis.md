# Stress Test 7: SHAP Value Analysis

## Overview
SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain the output of any machine learning model. It assigns each feature an importance value for a particular prediction.

## Why We Perform It
Black-box models like Deep Learning and XGBoost are hard to interpret. In financial systems, regulators (and bank analysts) demand explainability—we must know *why* a transaction was blocked. SHAP gives us a mathematically proven way to explain every single decision.

## How It Was Performed
We calculated global SHAP values across the test dataset to find the average impact of each feature on the model's magnitude of output.

## Detailed Results
| Feature | Mean |SHAP| |
|---------|--------------|
| `errorbalanceorig` | 3.9348 |
| `amt_to_bal_ratio` | 2.7744 |
| `balance_velocity` | 2.2822 |
| `ae_recon_error` | 1.4216 |

## Conclusion
The SHAP analysis perfectly corroborates the findings of the Feature Importance and AE Reliance tests. It confirms that the model's decision-making is multi-faceted. The Autoencoder's reconstruction error (`ae_recon_error`) has a meaningful SHAP value (1.42), proving it actively contributes to the decision, while features like `amt_to_bal_ratio` provide the heavy lifting when balance errors are unavailable.
