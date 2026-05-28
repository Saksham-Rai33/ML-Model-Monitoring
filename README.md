# Loan Approval Prediction using Machine Learning

## Project Overview

This project focuses on predicting whether a loan application will be approved or rejected using machine learning techniques. The objective is to build a robust classification system by applying multiple algorithms and comparing their performance.

The dataset includes financial and demographic information of applicants such as income, loan amount, credit score and asset values.

---

## Objectives

* Perform data preprocessing and feature engineering
* Train multiple machine learning models
* Evaluate and compare model performance
* Develop a reliable loan approval prediction system

---

## Dataset Description

The dataset consists of the following features:

* Number of Dependents
* Education
* Self Employed
* Annual Income
* Loan Amount
* Loan Term
* Credit Score
* Residential Asset Value
* Commercial Asset Value
* Luxury Asset Value
* Bank Asset Value
* Loan Status (Target Variable)

---

## Data Preprocessing

* Removed unnecessary columns such as `loan_id`
* Encoded categorical variables using Label Encoding
* Performed feature engineering:

  * Total Assets
  * Loan-to-Income Ratio
  * Asset-to-Loan Ratio
  * Income per Dependent
* Applied feature scaling using StandardScaler
* Split the dataset into training and testing sets (80/20)

---

## Models Used

### 1. XGBoost Classifier

XGBoost is a gradient boosting algorithm known for its high performance on structured data.

Key characteristics:

* Handles non-linear relationships effectively
* Includes regularization to reduce overfitting
* Optimized for speed and performance

---

### 2. Random Forest Classifier

Random Forest is an ensemble learning method based on multiple decision trees.

Key characteristics:

* Reduces overfitting through bagging
* Handles noisy data effectively
* Provides feature importance scores

---

### 3. Lasso Regression

Lasso Regression is a linear model that uses L1 regularization.

Key characteristics:

* Performs feature selection
* Produces interpretable models
* Controls overfitting in linear models

---

## Evaluation Metrics

The models were evaluated using the following metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

---

## Results Summary

| Model            | Accuracy | ROC-AUC  | F1 Score |
| ---------------- | -------- | -------- | -------- |
| XGBoost          | ~90%     | High     | High     |
| Random Forest    | ~88-91%  | High     | High     |
| Lasso Regression | ~80-85%  | Moderate | Moderate |

XGBoost achieved the best overall performance, followed by Random Forest. Lasso Regression provided interpretability but comparatively lower accuracy.

---

## Visualizations

The project includes the following visualizations:

* Confusion Matrix
* ROC Curve
* Precision-Recall Curve
* Feature Importance Plot
* Model Comparison Graphs

---

## Key Insights

* CIBIL score is a major factor influencing loan approval
* Income-to-loan ratio significantly impacts decisions
* Asset values contribute strongly to approval likelihood
* Ensemble models outperform linear models on this dataset

---

## Technology Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Matplotlib, Seaborn

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/loan-approval-ml.git
cd loan-approval-ml
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebooks:

```bash
jupyter notebook
```

---

## Future Improvements

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* Model deployment using Flask or Streamlit
* Incorporating additional real-world features
* Experimenting with advanced models

---

## Acknowledgment

This project was developed as part of a machine learning study to demonstrate practical implementation of classification models on financial data.
