
# 📘 Runbook Documentation for Update 2 Notebook

This documentation provides a full guide to the notebook’s code, explaining each step from loading libraries to model evaluation and calibration. Icons and placeholders indicate visuals and key sections.

---

## ⚙️ 1. Import Libraries

**Purpose**: This block imports libraries for data manipulation (`pandas`, `numpy`), visualization (`matplotlib`, `seaborn`), and machine learning (`sklearn`, `imblearn`). It also suppresses warnings to keep the output clean.

### Code:
```python
import pandas as pd
import os
from sys import platform
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve, f1_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, Lasso
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')
```

---

## 📂 2. Load Data

**Purpose**: Load the dataset and check its structure.

### Code:
```python
df = pd.read_csv("Telco_customer_churn.csv")
df.head()
```

### 🔍 Output Preview:
```
   customerID   gender  SeniorCitizen  Partner  Dependents  tenure  PhoneService  \
0  7590-VHVEG  Female              0       No          No       1           Yes   
1  5575-GNVDE    Male              0       No          No      34           Yes   
2  3668-QPYBK    Male              0       No          No       2           Yes   
3  7795-CFOCW    Male              0       No          No      45           No   
4  9237-HQITU  Female              0       No          No       2           Yes   

  MultipleLines InternetService OnlineSecurity ...
```

---

## 🔧 3. Pre-processing

### Inspect Column Types

**Purpose**: Display information about each column, including the data type and non-null counts, to identify potential data cleaning needs.

### Code:
```python
df.info()
```

#### 🔎 Output Example:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7043 entries, 0 to 7042
Data columns (total 21 columns):
 #   Column           Non-Null Count  Dtype 
---  ------           --------------  ----- 
 0   customerID       7043 non-null   object
 1   gender           7043 non-null   object
...
```

### Convert to Numeric

**Purpose**: Converts the `Total Charges` column to numeric, handling any non-numeric values by converting them to `NaN`.

### Code:
```python
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
```

---

### 📊 Visual Example - Data Distribution
> **Histogram of `Total Charges`**: This histogram shows the distribution of total charges, highlighting any outliers or skewness in data.

```python
# Example placeholder for visualization
plt.hist(df['Total Charges'].dropna(), bins=30)
plt.title("Total Charges Distribution")
plt.xlabel("Total Charges")
plt.ylabel("Frequency")
plt.show()
```

---

## 🧩 4. Exploratory Data Analysis (EDA)

**Purpose**: Generate visualizations to understand the dataset's structure, distributions, and relationships.

*Examples: histograms, box plots, and pair plots.*

### Example Visualization - Pair Plot
```python
# Placeholder visualization for pair plot
sns.pairplot(df[['Total Charges', 'Monthly Charges', 'tenure']], hue="Churn")
plt.show()
```

---

## 🛠️ 5. Feature Engineering

### Scaling

**Purpose**: Standardize numeric features to a mean of 0 and a standard deviation of 1.

### Code:
```python
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['NumericFeature1', 'NumericFeature2']])
```

---

### Handling Class Imbalance

**Purpose**: Use SMOTE to oversample the minority class.

### Code:
```python
smote = SMOTE()
X_res, y_res = smote.fit_resample(df_scaled, df['Target'])
```

---

## 🤖 6. Model Training

### Logistic Regression

**Purpose**: Train a logistic regression model with cross-validation.

### Code:
```python
model = LogisticRegressionCV(cv=5)
model.fit(X_train, y_train)
```

### Regularization Models (Ridge, Lasso)

**Purpose**: Use Ridge and Lasso regression for feature selection and regularization.

### Code:
```python
ridge = Ridge()
lasso = Lasso()
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
```

---

## 🏅 7. Model Evaluation

**Purpose**: Evaluate the model’s performance.

### Code:
```python
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred))
```

#### Sample Output:
```
Accuracy: 0.80
AUC-ROC: 0.82
```

---

## 🔧 8. Calibration and Analysis

**Purpose**: Calibrate model predictions.

### Code:
```python
prob_true, prob_pred = calibration_curve(y_test, model.predict_proba(X_test)[:, 1], n_bins=10)
plt.plot(prob_pred, prob_true)
plt.title('Calibration curve')
plt.xlabel('Predicted probability')
plt.ylabel('True probability')
plt.show()
```

#### Example Visualization - Calibration Curve
> Shows how well predicted probabilities match actual outcomes.

---

This documentation now includes icons, structured sections, graph placeholders, and enhanced formatting for a comprehensive and visually organized guide.
