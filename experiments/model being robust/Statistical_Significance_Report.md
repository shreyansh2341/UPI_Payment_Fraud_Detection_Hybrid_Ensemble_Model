# Statistical Significance Test (McNemar's Test)

## Objective
Determine whether the performance difference between the full V3 model (with leaky features) and the safe model (without `errorbalance*` features) is statistically significant.

## Methodology
Applied McNemar's test (with continuity correction) on the paired prediction outcomes of both models on the same test set. This tests whether the disagreements between models are symmetric (null hypothesis) or one model is systematically better.

## Results

| Model | Recall | Precision |
|---|---|---|
| Full V3 (with errorbalance) | 1.0000 | 0.9968 |
| Safe V3 (without errorbalance) | 0.9975 | 0.9680 |

| Statistic | Value |
|---|---|
| Discordant pairs (b) | 23 |
| Discordant pairs (c) | 132 |
| McNemar's chi-squared | 75.2516 |
| p-value | 0.000000 |

## Conclusion
The difference IS statistically significant (p=0.000000 < 0.05). The full model with `errorbalance` features performs measurably better. However, since the safe model still achieves 99.8% recall, the practical significance is minimal -- the model generalizes well even without the leaky features.
