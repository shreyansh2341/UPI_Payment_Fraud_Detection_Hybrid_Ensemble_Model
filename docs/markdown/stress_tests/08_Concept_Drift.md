# Stress Test 8: Concept Drift Analysis

## Overview
Concept Drift occurs when the statistical properties of the target variable (what the model is trying to predict) change over time. In fraud, this happens when fraudsters invent new scams that look completely different from historical data.

## Why We Perform It
A model that is 99% accurate in January might degrade to 70% accuracy by December if it cannot handle evolving patterns. Evaluating drift helps determine how often the model needs retraining.

## How It Was Performed
We divided the test dataset into 5 distinct chronological time bins and evaluated the recall and precision for each bin independently to ensure performance didn't degrade over time.

## Detailed Results
| Time Bin | Recall | Precision |
|----------|--------|-----------|
| Bin 1-5 | 1.0000 | ~1.0000 |

## Conclusion
The model maintained perfect stability across all time bins. However, we acknowledge an honest methodological limitation: the PaySim dataset is synthetic and its underlying generation rules do not fundamentally change over its 30-day simulated timeline. While the test confirms our model handles the dataset's entire temporal span flawlessly, real-world deployment would require continuous drift monitoring (e.g., using evidently.ai) to trigger automatic retraining.
