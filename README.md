# OmniEdge Customer Churn Command Center

AI-powered churn intelligence platform designed to predict customer attrition, quantify revenue at risk, and generate executive-ready retention strategies for subscription and recurring-revenue businesses.

This project demonstrates an end-to-end machine learning workflow that transforms raw customer records into production-style churn predictions, risk segmentation, retention action plans, and leadership-facing decision outputs.

---

## Executive Overview

Customer churn is one of the most persistent and expensive challenges in subscription-driven businesses. Most organizations identify churn after it happens, when the revenue loss has already occurred and recovery becomes significantly more expensive.

The OmniEdge Customer Churn Command Center was built to demonstrate how a modern retention intelligence system can move beyond reactive reporting and instead provide leadership teams with proactive decision support.

This platform enables stakeholders to answer high-value questions such as:

- Which customers are most likely to churn?
- How much annual revenue is currently at risk?
- Which model performs best for production use?
- Which customer segments should be prioritized first?
- What retention actions should be taken next?

---

## Business Problem

Recurring-revenue businesses often face three major challenges:

1. **Limited visibility into churn risk**
2. **Poor prioritization of retention resources**
3. **No clear link between model outputs and business action**

This project addresses those issues by combining machine learning, business logic, and executive reporting into a single retention decision system.

---

## Solution Summary

The platform combines:

- raw customer data ingestion
- schema validation and data profiling
- feature engineering for churn modeling
- multi-model machine learning evaluation
- full-customer scoring
- revenue-at-risk quantification
- retention action planning
- executive summary generation

The result is a consulting-grade churn intelligence system that translates predictive modeling into practical customer retention strategy.

---

## Core Capabilities

### 1. Customer Churn Prediction
Predicts churn probability for each customer using multiple machine learning models.

### 2. Model Benchmarking
Compares Logistic Regression, Random Forest, and XGBoost to identify the strongest production candidate.

### 3. Risk Segmentation
Classifies customers into operationally useful risk bands:

- Critical
- High
- Moderate
- Low

### 4. Revenue-at-Risk Analysis
Quantifies customer-level and portfolio-level financial exposure associated with churn.

### 5. Retention Action Planning
Generates targeted next-step actions based on customer risk, value, contract profile, and revenue impact.

### 6. Executive Decision Outputs
Produces leadership-ready summaries, retention playbooks, and analytical notebook outputs.

---

## Business Questions This Platform Answers

This system is designed to support executive, analytics, and customer-success decision-making around questions like:

- Which customers should the retention team contact first?
- Which risk segments represent the greatest revenue concentration?
- Which contract and pricing patterns are most associated with churn?
- Which predictive model should be operationalized?
- Which intervention types should be targeted to each customer segment?

---

## End-to-End Architecture

Raw Customer Data  
↓  
Schema Validation and Profiling  
↓  
Feature Engineering  
↓  
Model Training Pipeline  
↓  
Model Evaluation and Benchmarking  
↓  
Customer Scoring and Risk Segmentation  
↓  
Retention Action Engine  
↓  
Executive Summary Outputs  
↓  
Churn Command Center Analysis Notebook

---

## Repository Structure

```text
customer-churn-command-center/
│
├── config/
│   └── settings.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── churn_analytic_summary.xlsx
│   └── executive_insights.csv
│
├── logs/
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
│
├── notebooks/
│
├── scripts/
│   ├── 01_ingest_customer_data.py
│   ├── 02_validate_and_profile_data.py
│   ├── 03_feature_engineering.py
│   ├── 04_train_models.py
│   ├── 05_evaluate_models.py
│   ├── 06_score_customers.py
│   ├── 07_generate_retention_actions.py
│   └── 08_generate_executive_summary.py
│
├── churn_model_analysis.ipynb
├── requirements.txt
└── README.md
