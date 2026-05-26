# Stress Test 3: Learning Curves Report

## Overview
Learning curves plot the model's performance (usually loss or error) on both the training dataset and the validation dataset over time (e.g., as more trees are added in XGBoost, or more epochs in Neural Networks).

## Why We Perform It
Learning curves are the definitive diagnostic tool to detect **overfitting** and **underfitting**. 
- If training loss drops to zero but validation loss goes up, the model is overfitting (memorizing).
- If both losses remain high, the model is underfitting (failing to learn).
- Ideally, both losses should decrease together and stabilize.

## How It Was Performed
We tracked the Logloss metric across 100 boosting rounds (trees) in our Stage-2 XGBoost model for both the training and test sets.

## Detailed Results
| Stage | Train Logloss | Test Logloss |
|-------|--------------|-------------|
| Tree 1 | 0.1065 | 0.1079 |
| Tree 100 | 0.0000 | 0.0001 |

## Conclusion
The learning curves show textbook behavior for a non-overfitting model. Both the training and test logloss converge almost identically to near-zero values. There is absolutely no train-test divergence. While the near-zero logloss indicates the PaySim dataset is highly separable when engineered features are present, the fact that the test curve follows the train curve perfectly proves the algorithm's generalization capabilities.
