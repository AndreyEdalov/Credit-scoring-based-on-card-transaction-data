# Credit Scoring on Transaction Data with Reject Inference

## Project Overview

This project is dedicated to building a credit scoring model based on applicants’ transactional and financial data.  
A key methodological challenge addressed in the project is **selection bias**, which arises because default outcomes are fully observed only for approved applications, while rejected applicants have unknown credit behavior.

The project demonstrates a full end-to-end credit scoring workflow, including data analysis, feature validation, baseline modeling, reject inference, and comparison with alternative machine learning approaches.

---

## Objectives

- Build an interpretable **baseline credit scoring model**
- Identify and analyze **selection bias** in the data
- Apply **reject inference** to partially correct selection bias
- Compare baseline results with **alternative nonlinear models**
- Draw conclusions on model performance and practical trade-offs

---

## Dataset Description

The dataset contains information on credit card applications and includes the following key components:

### Application Outcome
- `APPL_OUTCM_CD`
  - `1` — approved application
  - `0` — rejected application

### Target Variable
- `TGT_VAR`
  - `1` — default
  - `0` — non-default

### Feature Groups
- Financial characteristics:
  - Annual income (`ANNUAL_INCOME_AMT`)
  - Household income (`APPL_PA_HHD_INC_AMT`)
- Behavioral indicators:
  - Short- and medium-term delinquency counts
  - Payment delay statistics
- External score:
  - Application score (`APPL_SCR_NO`)

Default outcomes are reliably observed only for approved applications, which leads to selection bias.

---

## Methodology

### 1. Exploratory Data Analysis

- Analyzed distributions of key numerical features
- Evaluated logical relationships between features and default rates
- Identified weak, non-monotonic, and noisy predictors

---

### 2. Baseline Model

- Built a **logistic regression model with WOE-transformed features**
- Performed feature selection using **Information Value (IV)**
- Trained the model using approved applications only
- Evaluated performance using ROC-AUC

**Baseline model performance:**  
ROC-AUC ≈ 0.53

---

### 3. Reject Inference

To address selection bias, a simple **hard-label reject inference** approach was applied:

- A baseline model was used to predict default probabilities for rejected applications
- Binary default labels were assigned based on predicted probabilities
- The training dataset was expanded to include both approved and rejected applications
- The logistic regression model was retrained on the augmented dataset

**Performance after reject inference:**  
ROC-AUC ≈ 0.54

---

### 4. Alternative Models

To assess whether nonlinear methods can improve predictive performance, two alternative models were tested:

#### Gradient Boosting
- Captures nonlinear relationships and feature interactions
- Achieved ROC-AUC ≈ 0.59

#### Decision Tree (with class balancing)
- Highlights threshold effects in key predictors
- Achieved ROC-AUC ≈ 0.60

In all models, income-related variables were the strongest drivers of credit risk.

---

## Results Summary

| Model                          | ROC-AUC |
|--------------------------------|---------|
| WOE Logistic Regression        | ~0.53   |
| Logistic + Reject Inference    | ~0.54   |
| Gradient Boosting              | ~0.59   |
| Decision Tree (Balanced)       | ~0.60   |

---

## Key Findings

- The dataset exhibits **strong selection bias**
- WOE-based logistic regression provides interpretability but limited performance
- Reject inference partially mitigates selection bias
- Nonlinear models significantly outperform the baseline
- Income is the dominant predictor of default risk across all models

---

## Conclusions

The project demonstrates the importance of accounting for selection bias in credit scoring tasks.  
While interpretable models remain essential for regulatory and business purposes, more flexible machine learning models can provide substantial performance improvements when interpretability constraints are relaxed.

---

## Technologies Used

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
