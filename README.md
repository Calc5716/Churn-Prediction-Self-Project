# 💡 Customer Churn Prediction Project

## Overview
This project implements an end-to-end Machine Learning pipeline to predict **customer churn** for a telecommunications company. The goal is to identify customers at high risk of leaving (`Churn=Yes`) using various behavioral and demographic features, allowing the business to proactively intervene with retention strategies.

The solution uses an **XGBoost Classifier**, optimized through **Optuna**, and deployed via a robust **imblearn Pipeline** that correctly handles data preprocessing and class imbalance.

---

## 🚀 Key Features and Techniques

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

---

## 🛠️ Corrected Pipeline Structure
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
The model is optimized for **ROC-AUC** and **F1-Score**, balancing the identification of churners (Recall) against minimizing false positives (Precision).

| Metric     | Result    | Interpretation |
|-----------|-----------|----------------|
| ROC-AUC   | XX.X%     | Probability the model ranks a random positive instance higher than a random negative instance. |
| F1-Score  | XX.X%     | Balanced measure of Precision and Recall. |
| Precision | XX.X%     | Of all customers predicted to churn, XX.X% actually churn. |
| Recall    | XX.X%     | XX.X% of all actual churners were correctly identified. |

> Replace `XX.X%` with actual scores from your final model evaluation.

---

## ⚙️ Dependencies
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `imbalanced-learn`
- `optuna`
- `ydata-profiling`
- `pickle`

---

## 💻 How to Run
1. Download the Jupyter Notebook: `Churn_Prediction_Project.ipynb`.
2. Ensure all dependencies are installed (use a **virtual environment** for best practice).
3. Open and execute the notebook cells sequentially in Jupyter.
4. The final, fitted pipeline is exported as a file (e.g., `final_churn_predictor_pipe.pkl`) and can be loaded for predictions in a production or scoring environment.
