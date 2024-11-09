# Runbook for Update2 Notebook

This runbook outlines each step, the purpose of each cell, and notes for running the cells in sequence.

---

### 1. Import Libraries
- **Purpose**: Load essential libraries for data manipulation, visualization, and machine learning.
- **Instructions**: Run this cell to ensure all required libraries are available. If any libraries are missing, use `pip install` to install them.

```python
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve, f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, Lasso
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')
```

### 2. Load Data
- **Purpose**: Load the primary dataset, likely related to customer churn.
- **Instructions**: Ensure the dataset file `Telco_customer_churn.csv` is in the same directory as the notebook. Run this cell to load the data.

```python
df = pd.read_csv('Telco_customer_churn.csv')
df.head()
```

### 3. Data Preprocessing
- **Purpose**: Prepare the data by converting column types, handling missing values, and selecting features for modeling.
- **Steps**:
  - **Inspect Column Types**: Run this cell to identify data types and spot columns requiring adjustments.
  - **Convert to Numeric**: Change columns like `Total Charges` to numeric format to handle non-numeric values.

```python
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df.info()
```

- **Drop Columns**: Remove unnecessary or redundant columns to streamline the model input.
```python
df.drop(columns=['UnneededColumn1', 'UnneededColumn2'], inplace=True)
```

### 4. Exploratory Data Analysis (EDA)
- **Purpose**: Generate visualizations to understand the distribution of target variables and key features.
- **Instructions**: Run each visualization cell in order, making adjustments based on observations if necessary.

### 5. Feature Engineering
- **Purpose**: Create or modify features, apply scaling, and handle class imbalance.
- **Steps**:
  - **Scaling**: Standardize numeric features using `StandardScaler`.
  - **Handling Class Imbalance**: Use SMOTE or Tomek Links to balance the target class.

```python
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['NumericFeature1', 'NumericFeature2']])
```

### 6. Model Training
- **Purpose**: Set up and train various machine learning models.
- **Steps**:
  - **Logistic Regression**: Train a logistic regression model, including a cross-validated version.
  - **Regularization Models**: Ridge and Lasso regression models.
  - **Hyperparameter Tuning**: Use `GridSearchCV` to find optimal parameters.

```python
model = LogisticRegressionCV(cv=5)
model.fit(X_train, y_train)
```

### 7. Model Evaluation
- **Purpose**: Assess model performance using metrics like accuracy, precision, recall, and AUC-ROC.
- **Instructions**: Run evaluation cells in sequence. Review confusion matrices and performance scores for model insights.

```python
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('AUC-ROC:', roc_auc_score(y_test, y_pred))
```

### 8. Calibration and Analysis
- **Purpose**: Calibrate the model, if necessary, to improve probability estimates.
- **Instructions**: Run the calibration curve cell to visualize calibration.

```python
prob_true, prob_pred = calibration_curve(y_test, model.predict_proba(X_test)[:, 1], n_bins=10)
plt.plot(prob_pred, prob_true)
```
