# Stress Test 4: Out-Of-Distribution (OOD) Stress Test

## Overview
An Out-Of-Distribution (OOD) test evaluates how the model handles data that looks fundamentally different from what it saw during training. It tests the boundaries of the model's knowledge.

## Why We Perform It
Fraudsters actively change their tactics. A model trained on 2025 fraud data might face entirely new (OOD) attacks in 2026. If the model relies purely on memorized historical patterns, it will fail against OOD data. This test ensures the model relies on fundamental behavioral anomalies rather than specific historical signatures.

## How It Was Performed
We created a strict OOD scenario by intentionally zeroing out the two most powerful features (`errorbalanceorig` and `errorbalancedest`). We neutralized them by setting them to 0 for all test cases, artificially crippling the model's primary detection mechanism. We then fed this "blinded" data to the model to see if it could still catch fraud.

## Detailed Results
| Condition | Recall |
|-----------|--------|
| Natural fraud (Baseline) | 98.10% |
| Neutralized leaky features | 96.98% |
| **Performance Drop** | **1.12%** |

## Conclusion
This is an exceptionally strong result. Even when the model's top features are completely stripped away (simulating a scenario where fraudsters figure out how to bypass our balance checks), the model's recall drops by only 1.12%. It successfully pivots to analyzing secondary indicators like velocity, transaction amount ratios, and temporal anomalies to identify the malicious behavior.
