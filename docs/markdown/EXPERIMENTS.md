# Experiments & Model Diversity

This document chronicles the experimental journey undertaken to build the current Fraud Detection System. We explored various architectures, data strategies, and diversity techniques to maximize performance.

## 1. Baseline vs. Stage 2 (Evolution)
*   **Baseline**: Initial experiments (`experiments/evaluate_baseline.py`) relied on simple logistic regression and single-tree methods. These suffered from high False Negative rates (missing frauds).
*   **Stage 2 (Current)**: We moved to advanced ensemble methods.
    *   **Improvement**: Introduced **XGBoost Stage 2** and **Random Forest Stage 2**.
    *   **Result**: Significant boost in Recall (Catching more fraud) without destroying Precision.

## 2. Model Diversity (The "Vast Diversity" Approach)
To ensure the model is not relying on a single "view" of the data, we implemented **Architecture Diversity**:

### A. Algorithmic Diversity
We purposefully chose models with different mathematical foundations:
1.  **Gradient Boosting (XGBoost)**: Low bias, learns from mistakes.
2.  **Bagging (Random Forest)**: Low variance, robust to noise.
3.  **Deep Learning (LSTM)**: Sequential/Temporal learning (Memory).
4.  **Unsupervised Learning (Autoencoder)**: Anomaly detection (Reconstruction Error).

**Why?** If one model fails (e.g., XGBoost overfits a specific pattern), the others (e.g., Autoencoder seeing an anomaly) can correct it.

### B. Feature Diversity
*   **Tabular Features**: Used by Tree models (Transaction amount, balances).
*   **Derived Features**: `errorBalance` (Engineered feature capturing math discrepancies in accounts).
*   **Sequential Features**: Used by LSTM (History of last N transactions).
*   **Latent Features**: Compressed representations from the Autoencoder.

## 3. Data Experiments
### A. Handling Imbalance
We experimented with multiple techniques to handle the massive 99.9% vs 0.1% imbalance:
*   **SMOTE (Synthetic Minority Over-sampling Technique)**: Generated synthetic fraud examples to train the baseline.
*   **Scale_Pos_Weight**: Used in XGBoost to mathematically weigh fraud classes higher during gradient descent.
*   **Focal Loss**: Implemented custom focal loss for LSTM to force the network to focus on "hard" examples rather than easy legitimate ones.

### B. Data Augmentation
*   **Experiment**: `experiments/stress_test_synthetic.py`
*   **Action**: Created `generate_synthetic_frauds.py` to simulate new, unseen fraud patterns.
*   **Goal**: To stress-test the model against attacks that haven't happened yet, ensuring future robustness.

## 4. Evaluation Strategy
We moved beyond simple "Accuracy" (which is misleading for fraud) to:
*   **PR-AUC (Precision-Recall Area Under Curve)**: The primary metric for success.
*   **Recall @ Precision**: Checking Recall specifically at high precision thresholds to ensure we don't block too many legitimate users.
