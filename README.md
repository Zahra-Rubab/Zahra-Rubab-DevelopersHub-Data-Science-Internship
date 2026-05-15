# DevelopersHub — Data Science & Analytics Internship
## Combined Tasks: Iris EDA, Customer Churn Prediction, Insurance Claim Prediction

**Intern:**  Zahra Rubab  
**Due Date:** 15th May, 2026

---

## Task 1: Exploring and Visualizing the Iris Dataset

### Objective
Understand how to read, summarize, and visualize a dataset using Python.

### Dataset
**Name:** Iris Dataset  
**Source:** Built-in Seaborn library — no download required  
**Size:** 150 rows × 5 columns  

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
4. Created 3 visualizations:
   - **Scatter Plot** → Sepal Length vs Sepal Width
   - **Histogram** → Distribution of all features
   - **Box Plot** → Petal Length by Species

### Results & Key Insights
- Dataset has **150 rows and 5 columns**
- **No missing values** found
- **Setosa** species has the smallest petal length — clearly separated from others
- **Virginica** has the largest sepal and petal measurements
- Petal length is the most useful feature to distinguish between species

### Visualizations
| Chart | Description |
|-------|-------------|
| task1_scatter.png | Sepal Length vs Sepal Width by species |
| task1_histogram.png | Distribution of all 4 numeric features |
| task1_boxplot.png | Petal Length spread across 3 species |

---

## Task 3: Customer Churn Prediction (Bank Customers)

### Objective
Identify customers who are likely to leave the bank using machine learning classification.

### Dataset
**Name:** Churn Modelling Dataset  
**Source:** Loaded directly using Python (no download required)  
**Size:** 1000 rows × 11 columns  

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
2. **Encoding:** 
   - Geography → Label Encoding (France=0, Germany=1, Spain=2)
   - Gender → Label Encoding (Female=0, Male=1)
3. **EDA:** Visualized churn count and age vs churn
4. **Model:** Trained Random Forest Classifier (100 trees)
5. **Evaluation:** Accuracy Score + Confusion Matrix
6. **Feature Importance:** Analyzed top factors driving churn

### Results & Key Insights
- Model Accuracy: **~86%**
- **Age** is the top predictor — older customers churn more
- **Balance** matters — high balance customers are more likely to leave
- **NumOfProducts** also strongly influences churn
- Random Forest performed well on both classes

### Visualizations
| Chart | Description |
|-------|-------------|
| task3_churn_count.png | Count of stayed vs left customers |
| task3_age_churn.png | Age distribution by churn status |
| task3_confusion_matrix.png | Model prediction accuracy matrix |
| task3_feature_importance.png | Top features influencing churn |

---

## Task 4: Predicting Insurance Claim Amounts

### Objective
Estimate medical insurance charges based on personal data using Linear Regression.

### Dataset
**Name:** Medical Cost Personal Dataset  
**Source:** Loaded directly from URL — no download required
