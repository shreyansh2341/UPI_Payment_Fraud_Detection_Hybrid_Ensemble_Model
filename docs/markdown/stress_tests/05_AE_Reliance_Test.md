# Stress Test 5: Autoencoder (AE) Reliance Test

## Overview
In our V5 Hybrid architecture, the Autoencoder (AE) serves as an unsupervised anomaly detector, feeding its `reconstruction_error` as a feature into the downstream XGBoost and Random Forest models. The AE Reliance Test evaluates what happens if the AE fails.

## Why We Perform It
In complex ensemble models, there is a risk of "lazy learning," where the final classifier simply outsources the entire decision to one sub-model. If the XGBoost model blindly trusts the Autoencoder, then any fraud that successfully tricks the Autoencoder will automatically trick the entire system. We need to verify that the components act independently.

## How It Was Performed
We performed an ablation study specifically on the Autoencoder. We replaced the legitimate `reconstruction_error` values with neutralized (average) values for all fraudulent transactions. We then measured if the XGBoost model failed to classify them.

## Detailed Results
| Condition | Recall |
|-----------|--------|
| Baseline (real AE errors) | 100.00% |
| Neutralized AE errors | 99.45% |
| **Performance Drop** | **0.55%** |

## Conclusion
The test proved that the XGBoost model uses the Autoencoder as a helpful hint, but not as a crutch. If the Autoencoder completely misses a fraudulent transaction (neutralized error), the primary classifier still successfully detects 99.45% of the fraud using the remaining behavioral tabular features. This demonstrates true defense-in-depth architecture.
