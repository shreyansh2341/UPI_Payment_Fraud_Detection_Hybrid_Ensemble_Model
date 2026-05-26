# Individual Model Analysis (V5 Hybrid Stack)

This document details the training, validation, and architectural methodologies for the individual models comprising the V5 Hybrid Fraud Detection System.

---

## 1. BiLSTM with Bahdanau Attention (Sequential/Temporal Model)
**Purpose**: Sequence modeling to capture temporal patterns, velocity attacks, and structuring (smurfing).

### Architecture
*   **Input**: Sequential transaction data (rolling windows of N time steps).
*   **Core**: Bi-directional LSTM layers to read the sequence forwards and backwards.
*   **Attention Mechanism**: Bahdanau Attention layer. This calculates context vectors and attention weights, allowing the model to mathematically focus on the specific timesteps within a window that are most indicative of fraud.
*   **Output**: Dense layers leading to a Sigmoid activation.

### Training Details
*   **Loss Function**: Focal Loss (`alpha=0.75`, `gamma=2.0`). This forces the model to focus on hard-to-classify fraudulent sequences rather than easily identifiable legitimate ones.
*   **Data Prep**: Upsampled to a 10% fraud ratio using Sequential SMOTE to ensure the temporal patterns of fraud were adequately represented.
*   **Vulnerability Addressed**: Advanced Stress Testing revealed a ~78% evasion rate against mathematical smurfing (adversarial perturbations). This is why the **Velocity Anti-Smurfing** heuristic is critical as a pre-check override.

---

## 2. Autoencoder (Unsupervised Feature Extractor)
**Purpose**: Anomaly detection via reconstruction error.

### Architecture
*   **Network**: Deep Neural Network Encoder (18->64->32->8) and Decoder (8->32->64->18).
*   **Training Data**: Trained **strictly on Normal (Legitimate) transactions**.
*   **Concept**: Because it only learns the latent representation of legitimate behavior, feeding it a fraudulent transaction results in a high Mean Squared Error (MSE) between input and output.
*   **Integration**: The calculated `reconstruction_error` is appended as a powerful new synthetic feature for the downstream supervised models.

---

## 3. XGBoost (Primary Supervised Classifier)
**Purpose**: High-precision tabular anomaly detection.

### Training Details
*   **Input**: Original Scaled Features + Engineered Velocity Features + Autoencoder Reconstruction Error.
*   **Imbalance Handling**: Uses `scale_pos_weight` (Ratio of Legit/Fraud, approx ~770) to heavily penalize missing fraud cases. It completely avoids traditional SMOTE to prevent synthetic noise in the primary decision path.
*   **Hyperparameters**: 
    *   `n_estimators`: 400-500
    *   `max_depth`: 4-6 (Shallow trees to prevent overfitting and ensure generalization).
    *   `learning_rate`: 0.05
*   **Evaluation**: "aucpr" (Area Under Precision-Recall Curve).
*   **Resilience**: Exhibited extreme robustness to Data Degradation, maintaining 88.7% recall even when 50% of the input features were Null (simulating catastrophic pipeline failure).

---

## 4. Random Forest (Variance Reducer)
**Purpose**: Robust ensemble partner to stabilize predictions.

### Training Details
*   **Input**: Same as XGBoost.
*   **Config**: 300 Trees, `max_depth=12` or unlimited with `min_samples_split=5`.
*   **Imbalance Handling**: `class_weight="balanced"`.
*   **Role**: Bagging reduces the variance from XGBoost's aggressive boosting, smoothing out borderline false positives.

---

## 5. Isolation Forest (Outlier Detector)
**Purpose**: Statistical outlier detection for the Path B anomaly tier.

### Training Details
*   **Config**: `n_estimators=200`, `contamination=0.001`.
*   **Role**: Complements the Autoencoder. While the AE looks for non-reconstructable patterns, the Isolation Forest isolates statistical outliers in the feature space. Used for secondary flagging of novel, zero-day attacks.
