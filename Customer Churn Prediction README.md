# Customer Churn Prediction Using Machine Learning

## Overview

This project predicts customer churn in a telecom company using supervised machine learning. Retaining existing customers is more cost-effective than acquiring new ones — this model identifies at-risk customers early so the business can take action.

Built as a midterm project for **CS-6470 (Machine Learning)** at Utah Valley University.

---

## Dataset

- **Source:** IBM Telco Customer Churn — Kaggle
- **Size:** \~7,000 customer records
- **Target variable:** Churn (Yes / No)
- **Class imbalance:** Only 19% of customers churned
- **Key features:** Monthly charges, tenure, internet service type, contract type

**Key findings from exploration:**

- Customers with higher monthly charges and longer tenure churn less
- Customers with no internet service churn more frequently
- Heavy internet users tend to stay

---

## Methods

### Preprocessing

- Label encoding for categorical variables
- Feature selection: dropped features with correlation > 0.8
- Dimensionality reduction with PCA
- Class imbalance handled with SMOTE (Synthetic Minority Oversampling Technique)
- Feature scaling with StandardScaler

### Models

- **Logistic Regression** — baseline + tuned with GridSearchCV (best C=0.1)
- **Random Forest Classifier** — feature importance + selection
- **XGBoost Classifier**
- **Ensemble blend** — Random Forest + XGBoost probabilities fed into Logistic Regression

---

## Results

| Pipeline | Accuracy |
| --- | --- |
| Random Forest + Logistic Regression | 81% |
| GridSearch-tuned Logistic Regression (C=0.1) | 80% |
| Ensemble (RF + XGB + LR blend) | 80% |
| PCA + SMOTE + Logistic Regression | 77% |

Dropping highly correlated features reduced accuracy from 81% to 77%, illustrating the trade-off between dimensionality reduction and predictive performance.

---

## How to Run

**1. Clone the repo**

```
git clone https://github.com/Luly7/customer-churn.git
cd customer-churn
```

**2. Install dependencies**

```
pip install -r requirements.txt
```

**3. Open the notebook**

```
jupyter notebook Midterm_Project_castleton.ipynb
```

**4. Run all cells** from top to bottom.

---

## Tech Stack

| Category | Libraries |
| --- | --- |
| Data | pandas, NumPy |
| Modeling | scikit-learn, XGBoost |
| Resampling | imbalanced-learn (SMOTE) |
| Visualization | matplotlib, seaborn |
| Environment | Python 3, Jupyter Notebook |

---

## Project Structure

```
customer-churn/
├── Midterm_Project_castleton.ipynb
├── requirements.txt
└── README.md
```
