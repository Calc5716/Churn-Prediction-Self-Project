# Customer Churn Prediction Project

## Overview
This project implements an end-to-end Machine Learning pipeline to predict **customer churn** for a telecommunications company. The goal is to identify customers at high risk of leaving (`Churn=Yes`) using various behavioral and demographic features, allowing the business to proactively intervene with retention strategies.

The solution uses an **XGBoost Classifier**, optimized through **Optuna**, and deployed via a robust **imblearn Pipeline** that correctly handles **data preprocessing, class imbalance, model optimization, and web deployment** through Flask, hosted on **Render**, providing users with an interactive web interface to predict churn probabilities in real time.

Users can input customer details through an intuitive interface built with HTML and CSS, and receive real-time predictions powered by the optimized XGBoost model.
(The Churn Web App codes are on the `master` branch).

---

## Dataset 

[Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn "Dataset")

---
## Web App Link

[Web App](https://churn-prediction-self-project.onrender.com/ "Visit Churn Prediction Website")

---

## Screenshots from the web App

1. ### Web App UI
![Dashboard Preview](images/Screenshot%202025-10-30%20164005.png)

2. ### Responsive UI for other devices
![Dashboard Preview](images/Screenshot%202025-10-30%20164317.png)

3. ### Output preview
![Dashboard Preview](images/Screenshot%202025-10-30%20164210.png)

---
## Key Features and Techniques

### Exploratory Data Analysis (EDA)
- In-depth analysis using **ydata-profiling** and custom visualizations.
- Understand feature distributions and correlations with churn.

### Data Preprocessing
- Handled by a single **ColumnTransformer** to prevent index misalignment and ensure consistency.
- **Imputation & Scaling:** `KNNImputer` and `StandardScaler` applied sequentially to numerical features (e.g., `TotalCharges`).
- **Encoding:** `OneHotEncoder` applied to categorical features.

### Class Imbalance Handling
- **SMOTE** (Synthetic Minority Over-sampling Technique) is integrated into the pipeline.
- Applied **only to preprocessed training data** to address imbalance in the target variable.

### Model Optimization
- **Optuna** used for advanced hyperparameter tuning of the XGBoost classifier.
- Performance metrics: **ROC-AUC** and **F1-Score**.

### Robust Deployment Pipeline
- A single consolidated **`imblearn.pipeline.Pipeline`** ensures:
  - All preprocessing steps, including SMOTE, are learned and applied correctly.
  - Prevents data leakage during training and prediction.
  - **Interactive Web Interface:** Built with **Flask**, **HTML**, and **CSS** for intuitive user interaction.  
-  **Cloud Deployment:** Hosted on **Render**, enabling live user access.  
-  **Environment Management:** Utilized a Python **virtual environment** with all dependencies listed in `requirements.txt`.  

---

## Pipeline Structure
The pipeline consolidates all preprocessing into a single `ColumnTransformer` before applying SMOTE and the final model.  
This structure is robust for **NumPy arrays** and **index-based transformations**.

**Pipeline Flow:**
1. **Preprocessor (`ColumnTransformer`)**  
   Applies all Imputation, Scaling, and Encoding based on original feature indices.
2. **SMOTE**  
   Resamples the transformed training data.
3. **Model (`XGBClassifier`)**  
   The fully optimized model.

---

## 📈 Results and Evaluation

### Logistic Regression Baseline Model
| Metric     | Result    |
|-----------|-----------|
| ROC-AUC   | 0.8537    | 
| F1-Score  | 0.7837     |
| Precision | 0.7551    | 
| Recall    | 0.8146     | 

### XGBoost Tuned Model
| Metric     | Result    |
|-----------|-----------|
| ROC-AUC   | 0.9316    | 
| F1-Score  | 0.8119     |
| Precision | 0.8378    | 
| Recall    | 0.8200     | 

---
## Tech Stack

- **Languages:** Python, HTML, CSS  
- **Frameworks/Libraries:** Flask, Scikit-learn, XGBoost, Optuna, imblearn, Pandas, NumPy  
- **Deployment:** Render  
- **Tools:** Git, VS Code, joblib, virtualenv 

---

## How It Works

1. **User Input:** Customer details are entered via the web form.  
2. **Backend Processing:** The Flask app loads the trained joblib pipeline and processes inputs.  
3. **Prediction:** The model outputs churn probability and classification result (e.g., “Likely to Churn”).  
4. **Display:** The result is displayed on the web interface dynamically.

---
