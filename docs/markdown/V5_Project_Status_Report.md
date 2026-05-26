# Project Status Report: V5 Hybrid Fraud Detection System

backend command- python -m uvicorn backend.app:app --port 8003 --host 127.0.0.1
frontend command- cd frontend
python -m streamlit run app.py --server.port 8501


## Executive Summary
This report provides a comprehensive overview of the **UPI-Based Fraud Detection System** developed to date. The project has evolved from baseline models into a highly sophisticated, production-ready **V5 Hybrid Deep Learning Ensemble**. It supports real-time, high-precision fraud detection across two distinct financial datasets: PaySim (Mobile Money/UPI transactions) and Credit Card transactions.

The system is characterized by its dual-path architecture (combining supervised auto-blocking with unsupervised anomaly flags) and the integration of temporal sequence modeling via BiLSTM/BiGRU networks equipped with Bahdanau Attention.

---

## 1. System Architecture Evolution

The project's architecture has progressed through several major iterations to address emerging fraud vectors.

### Previous Iterations
*   **Baseline & V1/V2:** Simple models (Logistic Regression, basic XGBoost), which struggled with high false-negative rates and class imbalances.
*   **V3 Hybrid:** Introduced a two-path architecture. 
    *   **Path A** used an XGBoost + Random Forest ensemble for immediate blocking. 
    *   **Path B** used an Autoencoder (AE) and Isolation Forest for unsupervised anomaly detection. 
    *   Achieved ~99.6% recall for known fraud patterns but lacked sequential context.
*   **V4 LSTM Integration:** Recognized that fraud often occurs in sequences (e.g., small tests followed by large cash-outs). Integrated BiLSTM/BiGRU models with Bahdanau Attention to catch temporal fraud. Rebalanced the data via sequential SMOTE.

### Current Architecture: V5 Hybrid
The V5 architecture is the culmination of these efforts, optimizing both structural and algorithmic diversity:

> [!NOTE] 
> The architecture evaluates transactions individually while retaining their temporal context.

**Dual-Path Flow:**
1.  **Path A (Auto-Block):** Highly precise supervised tier. Analyzes transactions using XGBoost and Random Forest. 
2.  **Path B (Flag for Review):** The anomaly/temporal tier. Uses Autoencoder reconstruction error, Isolation Forest outlier scores, and BiLSTM/Attention temporal anomaly scores to flag sophisticated evasion attempts (e.g., zero-day or step-money deduction attacks).

---

## 2. Machine Learning Stack

The V5 intelligence layer leverages varied approaches ("Vast Diversity"):

*   **XGBoost:** Primary classifier for high-precision tabular detection. Tuned with `scale_pos_weight` to handle severe class imbalance without traditional SMOTE.
*   **Random Forest:** Acts as a variance reducer to smooth XGBoost predictions and prevent overfitting.
*   **Autoencoder:** Trained strictly on legitimate transactions. It flags anomalies based on high reconstruction errors, acting as a crucial unsupervised feature signal.
*   **BiLSTM with Bahdanau Attention:** Examines a window of transactions to detect velocity attacks and sequencing anomalies. Bahdanau attention adds explainability, weighting the exact timesteps that indicate fraud.

---

## 3. Data Pipelines & Engineering

The system processes two main datasets:
1.  **PaySim (Main Focus):** ~6.35 million simulated mobile money transactions. Fraud rate: ~0.13%. 
2.  **Credit Cards:** ~284,000 transactions. Fraud rate: ~0.17%.

### Feature Engineering Highlights
*   **Balance Error Checking:** Engineered `errorBalanceOrig` and `errorBalanceDest` to capture mathematical inconsistencies often caused by fraudsters bypassing standard app logic.
*   **Velocity Features:** Calculated cumulative amounts, transaction counts, and time-since-last to provide inputs for the sequential models.
*   SMOTE was selectively used (e.g., to upsample sequence data to 10% fraud for BiLSTM training) while primary models relied on focal loss and class weighting.

---

## 4. Application Interfaces

The model is deployed with a microservices-inspired approach:

*   **Backend (FastAPI, Port 8003):** An asynchronous, high-performance REST API.
    *   **Endpoints:** Real-time `/predict` endpoints using Pydantic schemas for data validation.
    *   **Smart Parsing:** Automatically detects CSV structural formats (raw PaySim, engineered full, legacy, or credit card).
    *   **Lazy Loading:** Models are cached in memory via `model_loader.py` to ensure latency remains under 50ms.
*   **Frontend (Streamlit, Port 8501):** An interactive, real-time batch processing dashboard.
    *   Users can upload CSVs and view a live-updating progress bar.
    *   Results are color-coded (Red for Fraud, Green for Legit) and include summary metrics.

*Additional UI work*: Recent conversations indicate refinements to user order history and admin cancellation request interfaces in a broader React-based system (possibly an integrated admin portal).

---

## 5. Recent Work & Verifications

Based on recent context, the latest enhancements include:
1.  **"Step Money Deduction" Attack Simulation:** Verified the V5 hybrid model against novel zero-day attack patterns that bypassed V3. Path B (BiLSTM) successfully handled these.
2.  **Documentation Finalization:** Updated the `Major_Project_Report.docx` and produced native, editable PPT/Word diagrams (System Architecture, DFDs, Flowcharts, Use Cases) reflecting V5 accurately.
3.  **UI/UX Refinements:** Fixed bugs related to order cancellation visibility, implemented React-hot-toast notifications, and added visual "New" indicators for admin dashboards.

---

## Next Steps

The theoretical machine learning foundation is rock-solid and the system has been tested against novel attacks. To proceed further, potential tasks might include:

1.  **UI/UX App Integration:** Continuing to integrate the UI components (like the Admin cancellation dashboard) with the backend processing logic.
2.  **Stitch MCP Exploration:** The recent interest in **Stitch MCP** suggests you might want to start applying specialized design systems, creating new React/web application screens, or altering frontend themes based on the fraud detection admin panel.
3.  **Code Optimization / Refactoring:** Doing a final pass on `src/` modularity, cleaning up unused V3/V4 legacy files if they are no longer required.
4.  **Deployment Prep:** Containerizing the whole stack (Docker) for cloud deployment.
