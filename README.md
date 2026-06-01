# Credit Risk Probability Model

## Credit Scoring Business Understanding

### Basel II and Model Interpretability

Basel II requires financial institutions to develop transparent and well-documented risk models. Credit decisions must be explainable to regulators and internal stakeholders. Therefore, model interpretability, documentation, monitoring, and reproducibility are critical requirements.

### Need for a Proxy Variable

The Xente dataset contains transaction information but does not contain an explicit default label. A proxy variable is therefore required to approximate customer credit risk. We will use customer engagement behavior through RFM analysis to identify potentially high-risk customers.

### Risks of Proxy-Based Prediction

Proxy variables do not represent actual loan defaults. This introduces risks such as incorrect labeling and prediction bias. Results should therefore be interpreted as estimated risk rather than verified default behavior.

### Interpretable vs High-Performance Models

Logistic Regression combined with Weight of Evidence (WoE) offers transparency and regulatory friendliness. Ensemble methods such as Random Forest and XGBoost may provide better predictive performance but are more difficult to explain. Both approaches will be evaluated.