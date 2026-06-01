# Credit Risk Probability Model for Alternative Data

## Project Overview

This project develops an end-to-end credit risk scoring system for Bati Bank using transaction data from the Xente eCommerce platform. Since the dataset does not contain an actual loan default label, a proxy target variable is created using RFM (Recency, Frequency, Monetary) customer behavior analysis.

The final solution includes:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Proxy Target Variable Construction
- Machine Learning Model Training
- MLflow Experiment Tracking
- FastAPI Model Deployment
- Docker Containerization
- CI/CD Automation using GitHub Actions

## Credit Scoring Business Understanding

### Basel II and Model Interpretability

The Basel II Accord emphasizes accurate risk measurement, transparency, and documentation in credit risk modeling. Financial institutions must be able to explain how a credit decision was reached and demonstrate that the model is reliable and monitored over time. Therefore, model interpretability and reproducibility are critical considerations.

### Need for a Proxy Variable

The Xente dataset does not contain a direct loan default indicator. Since supervised machine learning requires labeled outcomes, a proxy target variable must be created. Customer behavioral patterns derived from RFM analysis are used to identify potentially high-risk customers.

### Risks of Proxy-Based Prediction

A proxy variable is an approximation rather than a true measure of default. This may introduce labeling errors where some low-risk customers are classified as high-risk and vice versa. The resulting model predicts behavioral risk rather than actual repayment behavior.

### Trade-Off Between Interpretable and High-Performance Models

Logistic Regression combined with Weight of Evidence (WoE) provides strong interpretability and regulatory compliance. More complex models such as Random Forest and XGBoost often achieve higher predictive performance but may be harder to explain. Both model families will be evaluated and compared.
