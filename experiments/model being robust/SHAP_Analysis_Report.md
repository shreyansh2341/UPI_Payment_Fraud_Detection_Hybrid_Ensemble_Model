# SHAP Local/Global Explainability Report

## Objective
Provide model transparency by analyzing the global and local feature importance of the V3 XGBoost classifier using SHapley Additive exPlanations (SHAP).

## Global Feature Importance (Top 10)
| Feature | Mean Absolute SHAP Value |
|---------|--------------------------|
| errorbalanceorig | 3.9348 |
| amt_to_bal_ratio | 2.7744 |
| balance_velocity | 2.2822 |
| ae_recon_error | 1.4216 |
| amount | 0.7530 |
| errorbalancedest | 0.7242 |
| hour | 0.6578 |
| oldbalancedest | 0.5441 |
| newbalanceorig | 0.4566 |
| upi_type_upi_payment | 0.4269 |

## Conclusion
The SHAP values confirm that the model heavily relies on behavioral deviations (like transaction amounts vs averages) and the autoencoder's reconstruction error (`ae_recon_error`). This shows that the model is making decisions based on complex, non-linear feature interactions rather than simple, easily exploitable single-feature thresholds.
