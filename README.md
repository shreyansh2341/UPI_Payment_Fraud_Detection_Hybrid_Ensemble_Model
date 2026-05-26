# UPI Payment Fraud Detection Hybrid Ensemble Model

This project aims to monitor and mitigate the risk of UPI Frauds by analyzing payments with the help of a two-path hybrid architecture. It utilizes multiple supervised, unsupervised, and deep learning models including Random Forest, XGBoost, Autoencoders, Isolation Forest, and BiLSTM with Bahdanau Attention.

## 📖 Project Documentation

This repository contains comprehensive documentation detailing every aspect of the project, from data cleaning to model architectures and stress testing. Please refer to the following markdown files for in-depth information:

### 🏗️ Architecture & Walkthrough
*   [**Project Walkthrough**](docs/markdown/PROJECT_WALKTHROUGH.md): A complete guide and overview of the project's workflow.
*   [**Architecture Details**](docs/markdown/ARCHITECTURE.md): Deep dive into the system design and the two-path hybrid architecture.
*   [**V5 Project Status Report**](docs/markdown/V5_Project_Status_Report.md): Current status and overview of the V5 hybrid inference pipeline.

### 📊 Data & Preprocessing
*   [**Data Cleaning**](docs/markdown/DATA_CLEANING.md): Methodologies and steps for cleaning the raw transaction data.
*   [**Dataset & SMOTE**](docs/markdown/DATASET_AND_SMOTE.md): Details about the dataset distribution and how SMOTE was used to handle class imbalance.

### 🧠 Models & Performance
*   [**Model Stack**](docs/markdown/MODEL_STACK.md): Overview of the entire ensemble model stack.
*   [**Individual Models**](docs/markdown/MODELS_INDIVIDUAL.md): Detailed information on individual algorithms used (Random Forest, XGBoost, Autoencoders, Isolation Forest).
*   [**Model Comparison**](docs/markdown/MODEL_COMPARISON.md): Comparative analysis between different models.
*   [**V4 LSTM Integration**](docs/markdown/V4_LSTM_INTEGRATION.md): Specifics on integrating the BiLSTM with Bahdanau Attention.
*   [**Model Performance (V5)**](docs/markdown/MODEL_PERFORMANCE_V5.md): The latest performance metrics and evaluation results.
*   [**Experiments**](docs/markdown/EXPERIMENTS.md): Log of experiments and hyperparameters tested.

### 🛡️ Robustness & Stress Testing
*   [**Complete Robustness Audit**](docs/markdown/Complete_Robustness_Audit.md): Comprehensive audit of the model's robustness against various vulnerabilities.
*   [**Stress Tests Directory**](docs/markdown/stress_tests/): A collection of individual stress test reports analyzing concept drift, noise guard robustness, and demographic fairness parity.

## 🚀 Project Structure

*   `src/`: Contains all the source code, training scripts, and inference logic.
*   `docs/`: Contains all documentation, images, and architecture diagrams.
*   `notebooks/`: Jupyter notebooks used for initial EDA and model prototyping.
*   `models/`: Directory where trained model artifacts are stored.
