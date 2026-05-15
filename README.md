# Data Science & Analytics Internship — Task Summary
### DevelopersHub Corporation

**Intern:**  Zahra Rubab  
**Due Date:** 15th May, 2026

---

## Overview

This document covers three data science tasks completed as part of the DevelopersHub Corporation internship, spanning exploratory data analysis, classification, and regression.

| Task | Title | Technique |
|------|-------|-----------|
| Task 1 | Exploring and Visualizing the Iris Dataset | EDA & Visualization |
| Task 3 | Customer Churn Prediction (Bank Customers) | Random Forest Classification |
| Task 4 | Predicting Insurance Claim Amounts | Linear Regression |

---

## Task 1: Exploring and Visualizing the Iris Dataset

### Objective
Understand how to read, summarize, and visualize a dataset using Python.

### Dataset
**Name:** Iris Dataset — Built-in Seaborn library (150 rows × 5 columns)

| Column | Description |
|--------|-------------|
| sepal_length | Length of sepal in cm |
| sepal_width | Width of sepal in cm |
| petal_length | Length of petal in cm |
| petal_width | Width of petal in cm |
| species | Type of flower (Setosa, Versicolor, Virginica) |

### Approach
1. Loaded dataset using `sns.load_dataset('iris')`
2. Explored structure using `.shape`, `.columns`, `.head()`, `.describe()`
3. Checked for missing values
4. Created 3 visualizations: Scatter Plot, Histogram, Box Plot

### Key Insights
- Dataset has **150 rows and 5 columns** with **no missing values**
- **Setosa** has the smallest petal length — clearly separated from other species
- **Virginica** has the largest sepal and petal measurements
- Petal length is the most useful feature to distinguish between species

### Visualizations
| File | Description |
|------|-------------|
| task1_scatter.png | Sepal Length vs Sepal Width by species |
| task1_histogram.png | Distribution of all 4 numeric features |
| task1_boxplot.png | Petal Length spread across 3 species |

### Libraries
```python
pandas, matplotlib, seaborn
```

### How to Run
```bash
pip install pandas matplotlib seaborn
```
Open `Task1_Iris_EDA.ipynb` in Jupyter Notebook and run **Kernel → Restart & Run All**.

---

## Task 3: Customer Churn Prediction (Bank Customers)

### Objective
Identify customers who are likely to leave the bank using machine learning classification.

### Dataset
**Name:** Churn Modelling Dataset (1000 rows × 11 columns)

| Column | Description |
|--------|-------------|
| CreditScore | Customer's credit score |
| Geography | Country (France, Germany, Spain) |
| Gender | Male / Female |
| Age | Customer's age |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Has credit card (1=Yes, 0=No) |
| IsActiveMember | Active member (1=Yes, 0=No) |
| EstimatedSalary | Customer's estimated salary |
| Exited | **Target:** 1=Left bank, 0=Stayed |

### Approach
1. **Data Cleaning:** Removed unnecessary columns
2. **Encoding:** Label Encoding for Geography and Gender
3. **EDA:** Visualized churn count and age vs churn
4. **Model:** Random Forest Classifier (100 trees)
5. **Evaluation:** Accuracy Score + Confusion Matrix
6. **Feature Importance:** Analyzed top factors driving churn

### Key Insights
- Model Accuracy: **~86%**
- **Age** is the top predictor — older customers churn more
- **Balance** matters — high balance customers are more likely to leave
- **NumOfProducts** also strongly influences churn

### Visualizations
| File | Description |
|------|-------------|
| task3_churn_count.png | Count of stayed vs left customers |
| task3_age_churn.png | Age distribution by churn status |
| task3_confusion_matrix.png | Model prediction accuracy matrix |
| task3_feature_importance.png | Top features influencing churn |

### Libraries
```python
pandas, matplotlib, seaborn, scikit-learn
```

### How to Run
```bash
pip install pandas matplotlib seaborn scikit-learn
```
Open `Task3_Churn_Prediction.ipynb` in Jupyter Notebook (internet required) and run **Kernel → Restart & Run All**.

---

## Task 4: Predicting Insurance Claim Amounts

### Objective
Estimate medical insurance charges based on personal data using Linear Regression.

### Dataset
**Name:** Medical Cost Personal Dataset — loaded from GitHub URL (1338 rows × 7 columns)

| Column | Description |
|--------|-------------|
| age | Age of the person |
| sex | Gender (male/female) |
| bmi | Body Mass Index |
| children | Number of children |
| smoker | Smoker or not (yes/no) |
| region | Residential area (northeast, southeast, etc.) |
| charges | **Target:** Medical insurance charges in USD |

### Approach
1. **Data Loading:** Loaded from GitHub URL
2. **Encoding:** Label encoding for sex/smoker; One-Hot Encoding for region
3. **EDA:** BMI vs Charges, Age vs Charges, Smoker vs Charges visualizations
4. **Model:** Linear Regression
5. **Evaluation:** MAE and RMSE
6. **Visualization:** Actual vs Predicted scatter plot

### Key Insights
- **Smoking** is the strongest predictor of high charges
- Smokers pay **3–4x more** than non-smokers
- **Age** positively correlates with charges
- **Higher BMI** leads to higher insurance costs

### Evaluation Metrics
| Metric | Description |
|--------|-------------|
| MAE (Mean Absolute Error) | Average prediction error in USD |
| RMSE (Root Mean Squared Error) | Penalizes large errors more |

### Visualizations
| File | Description |
|------|-------------|
| task4_bmi_charges.png | BMI vs Charges colored by smoker |
| task4_age_charges.png | Age vs Charges colored by smoker |
| task4_smoker_charges.png | Smoker vs Non-smoker charges |
| task4_actual_vs_predicted.png | Model accuracy visualization |

### Libraries
```python
pandas, matplotlib, seaborn, scikit-learn, numpy
```

### How to Run
```bash
pip install pandas matplotlib seaborn scikit-learn numpy
```
Open `Task4_Insurance_Prediction.ipynb` in Jupyter Notebook (internet required) and run **Kernel → Restart & Run All**.

---

## Common Setup

All tasks use Python with Jupyter Notebook. To install all dependencies at once:

```bash
pip install pandas matplotlib seaborn scikit-learn numpy
```

---

*DevelopersHub Corporation — Data Science & Analytics Internship*
