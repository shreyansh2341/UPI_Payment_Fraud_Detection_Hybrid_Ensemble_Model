# Concept Drift Evaluation Report

## Objective
Test the model's sensitivity to time decay by evaluating performance across distinct temporal splits in the test data.

## Results Across Time Bins
| Time Bin | Recall | Precision |
|----------|--------|-----------|
| Bin_1 | 1.0000 | 1.0000 |
| Bin_2 | 1.0000 | 0.9143 |
| Bin_3 | 1.0000 | 1.0000 |
| Bin_4 | 1.0000 | 1.0000 |
| Bin_5 | 1.0000 | 0.9980 |

## Conclusion
The performance remains stable across the different time bins within the test set. There is no significant decay in recall or precision, indicating that the engineered features (like velocity and relative amounts) provide robust signals that do not degrade rapidly as time progresses.
