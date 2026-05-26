# T15: Probability Calibration Analysis

## Overview
Tree-based ensembles (like Random Forest and XGBoost) and models trained on heavily resampled datasets (like SMOTE) often produce miscalibrated probabilities. A predicted score of 0.8 does not necessarily mean an 80% chance of fraud. This test evaluates the Expected Calibration Error (ECE) before and after applying the sharpening mitigation.

## Methodology
- Calculated baseline Expected Calibration Error (ECE) using 10 uniform bins.
- Applied the `sharpen_probabilities` transformation.
- Recalculated ECE to quantify the improvement.

## Results
* **ECE Raw (Baseline)**: 0.3463
* **ECE Sharpened**: 0.0320
* **Relative Improvement**: 31.43% (Absolute reduction in ECE)

### Calibration Bins (Sharpened)
* Mean Predicted vs. True Fraction of Positives aligns very closely at the extremes:
  * Bin 1: Mean Pred = 0.053, True Pos = 0.085
  * Bin 10: Mean Pred = 0.914, True Pos = 1.000

## Analysis & Conclusion
The raw model outputs were severely miscalibrated (ECE ~34.6%), pulling probabilities toward the center. Applying the sharpening calibration routine successfully pushed the probabilities back toward the extremes (0 and 1), drastically reducing the Expected Calibration Error to an excellent 3.2%.

This mitigation is critical for the routing logic. Because V5 relies on strict thresholding to route transactions to Block vs. Review, well-calibrated probabilities ensure that high-confidence blocks are genuinely high-confidence, minimizing false-positive friction for end users.
