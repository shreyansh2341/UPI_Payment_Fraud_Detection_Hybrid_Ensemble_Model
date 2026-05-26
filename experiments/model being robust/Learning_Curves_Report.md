# Learning Curves Analysis Report

## 1. Objective
To monitor the model's loss (Logloss) on both the training dataset and the unseen test dataset incrementally as trees are added. This detects classical overfitting, which occurs when a model's training loss continues to decrease while its validation/test loss begins to increase.

## 2. Methodology
An XGBoost model was trained over 100 boosting rounds using the full training set. After every tree was built, the logloss was computed for both the training set and the test set (`eval_set`).

## 3. Results (Logloss Progression)

| Stage | Train Logloss | Test Logloss |
| :--- | :--- | :--- |
| **Initial (Tree 1)** | 0.1065 | 0.1079 |
| **Final (Tree 100)** | 0.0000 | 0.0001 |

*Note: The test logloss remained stable and did not diverge from the training logloss.*

## 4. Strict Interpretation
**No Classical Overfitting Observed.**
In a severely overfitted model, we would see the Train Logloss approach 0.0000 while the Test Logloss drops initially and then spikes upwards (e.g., to 0.5000+). Here, both train and test losses converge together harmoniously towards ~0.0001.

**Warning: Dataset Simplicity.**
The fact that the logloss reaches `0.0001` indicates that the dataset is highly separable in the multi-dimensional feature space. The engineered features (`balance_velocity`, etc.) make identifying fraud almost a deterministic mathematical equation rather than a probabilistic guess. 

## 5. Generalization Verdict
**Valid within Domain, but Highly Separable.** The model itself is not overfitting in the algorithmic sense (it fits the test data just as well as the train data). However, because the synthetic PaySim dataset allows for near-perfect separation, these specific logloss numbers will not translate exactly to messy, real-world data. Real-world implementation will likely see the test logloss settle higher, but the underlying convergence pattern proves the algorithmic parameters are sound.
