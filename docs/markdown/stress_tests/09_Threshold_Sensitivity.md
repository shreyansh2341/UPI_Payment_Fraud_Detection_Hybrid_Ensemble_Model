# Stress Test 9: Threshold Sensitivity Analysis

## Overview
Machine learning models don't output binary "Fraud" or "Legit" decisions; they output a probability (e.g., 0.85). The Threshold is the cutoff point (e.g., 0.50). Anything above is Fraud, anything below is Legit. Threshold Sensitivity evaluates how changing this cutoff impacts the model.

## Why We Perform It
In fraud detection, different business scenarios require different thresholds. A high-value wire transfer might use a low threshold (0.25) to catch any slight suspicion, while a small coffee purchase might use a high threshold (0.80) to avoid declining a customer's card. The model must remain stable across different cutoffs.

## How It Was Performed
We evaluated the Precision, Recall, and F1-score of the ensemble model across a range of thresholds from 0.10 to 0.95.

## Detailed Results
| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| 0.10 | 0.9662 | 1.0000 | 0.9828 |
| 0.50 | 0.9968 | 1.0000 | 0.9984 |
| 0.90 | 1.0000 | 1.0000 | 1.0000 |

## Conclusion
The model demonstrates remarkable threshold stability. The fact that recall remains at 100% even at highly strict thresholds (0.90) indicates that when the model detects fraud, it does so with extreme confidence (outputting probabilities near 0.99 rather than borderline 0.51). This allows the business to safely set high thresholds to minimize false positive friction for customers without sacrificing security.
