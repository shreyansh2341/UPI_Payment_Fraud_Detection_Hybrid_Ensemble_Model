# Dataset Description & SMOTE Integration

## 1. Introduction for the Team
Hello team! This document serves as a comprehensive guide to the datasets used in our Fraud Detection project and how we handle the critical challenge of extreme class imbalance using SMOTE. Understanding our data foundation is crucial for grasping why our V5 Hybrid architecture behaves the way it does.

## 2. PaySim Dataset Overview
The primary dataset we are using is the **PaySim dataset**, which is a synthetic financial dataset. It is specifically designed to mimic real-world mobile money transactions (very similar to India's UPI system or platforms like Venmo/M-Pesa).

| Property | Value |
|:---------|:------|
| **Total Transactions** | 6,354,407 |
| **Fraudulent Transactions** | 8,213 (0.13%) |
| **Legitimate Transactions** | 6,346,194 (99.87%) |
| **Time Span** | 744 hours (~31 days) |
| **Transaction Types** | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| **Fraud-eligible Types** | TRANSFER, CASH_OUT only |

### The Class Imbalance Problem
As you can see, the dataset exhibits extreme class imbalance with only **0.13% fraud**. 
* **Why does this happen?** This is actually a highly realistic scenario. In the real world, fraud rates are typically between 0.1% and 0.5%. 
* **Why is it a problem?** If we feed this directly to a machine learning model, the model will likely learn to just predict "Legitimate" for every transaction. It will achieve 99.87% accuracy, but it will have a 0% recall (catching 0 frauds). It fails to learn the intricate, hidden patterns of fraud. Therefore, we need specialized sampling techniques.

## 3. Data Split Strategy

Before addressing the imbalance, we must split our data. We use a **stratified split** to preserve the 0.13% fraud ratio across all subsets:

| Split | Purpose | Size | Fraud Cases | Fraud % |
|:------|:--------|:-----|:------------|:--------|
| **Training** (70%) | Model learning | 4,448,085 | 5,749 | 0.13% |
| **Validation** (15%) | Threshold tuning | 953,161 | 1,232 | 0.13% |
| **Test** (15%) | Final evaluation & metrics | 953,161 | 1,232 | 0.13% |

**Why stratified?** Random splitting could result in a validation or test set with zero fraud samples. Stratification guarantees proportional fraud representation in every split, ensuring our evaluations are statistically valid.

## 4. SMOTE Oversampling

### What is SMOTE?
SMOTE stands for **Synthetic Minority Oversampling Technique**. Instead of simply duplicating existing fraud records (which leads to severe overfitting), SMOTE mathematically generates *new, synthetic* fraud samples by interpolating between existing fraud data points in the feature space.

### Application Rules
**CRITICAL RULE:** SMOTE is applied **ONLY to the training set**. 
If we applied it before splitting, synthetic data would leak into the test set, creating an illusion of high performance (Data Leakage). 

| Metric | Before SMOTE | After SMOTE |
|:-------|:-------------|:------------|
| Legitimate samples | 4,442,336 | 4,442,336 (unchanged) |
| Fraud samples | 5,749 | ~444,234 |
| Fraud ratio | 0.13% | **~10%** |
| Total training size | 4,448,085 | ~4,886,570 |

### Why a 10% target ratio?
You might ask, "Why not balance it to 50/50?"
1. **Overfitting risk:** Boosting 5,749 samples to 4.4 million (50/50) would create a massive amount of synthetic, repetitive noise.
2. **Signal distortion:** 10% provides the models with enough mathematical "signal" to learn fraud boundaries without completely distorting the natural probability distribution of the data. 

### Where SMOTE is NOT applied:
- **Validation set**: Kept original for unbiased threshold tuning.
- **Test set**: Kept original to reflect real-world distributions for honest evaluation.
- **Autoencoder training**: The AE is trained exclusively on legitimate transactions to learn normal patterns, so it does not see SMOTE data.

## 5. Model-Specific Training Strategies

Because our V5 architecture is a hybrid ensemble, different models handle the data differently:

| Model | Training Data | Imbalance Handling |
|:------|:-------------|:-------------------|
| **XGBoost (Path A)** | SMOTE-augmented train | `scale_pos_weight` parameter + SMOTE |
| **Random Forest (Path A)** | SMOTE-augmented train | `class_weight='balanced'` + SMOTE |
| **Autoencoder (Path B)** | Legitimate-only train | Learns normal behavior patterns; anomaly detected via reconstruction error |
| **BiLSTM (Path B)** | SMOTE-augmented sequences| Learns temporal patterns across padded sequences |

## 6. Feature Scaling

After SMOTE, we apply **StandardScaler**:
- **Fit**: Computed on the training data only.
- **Transform**: Applied to the validation and test sets.
- **Purpose**: Ensures zero mean and unit variance for all features. This is mathematically critical for deep learning models (Autoencoder, BiLSTM) to converge quickly and prevents features with large absolute values (like transaction amounts) from dominating the loss gradients.
