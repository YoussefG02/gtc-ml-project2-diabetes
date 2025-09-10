# Diabetes Prediction

## Project Overview

This project focuses on building and evaluating machine learning models to **predict the onset of diabetes**.
The objective is to implement a complete data science workflow, from initial data exploration and preparation to model training, hyperparameter tuning, and final evaluation.
The analysis uses the **Pima Indians Diabetes Database**.

-----

## Repository Contents

  - **`Diabetes.ipynb`** → Jupyter Notebook containing the entire workflow (EDA → Data Prep → Model Building → Prediction).
  - **`diabetes.csv`** → Raw dataset used for the analysis.
  - **`README.md`** → Project documentation (this file).

-----

## Workflow

### Phase 1: Data Exploration & EDA

  - Generated summary statistics and dataset info to get a high-level overview.
  - Checked for missing values and found none in the dataset.
  - Detected outliers using the **Z-score method** (`|Z| > 3`) and identified several in columns like `BloodPressure`, `Insulin`, and `BMI`.
  - Visualized key aspects of the data:
      - Plotted the class balance for the `Outcome` variable.
      - Used **KDE plots** to compare the distributions of `Glucose`, `BMI`, and `Age` for diabetic vs. non-diabetic outcomes.
      - Created a **scatterplot** to analyze the relationship between `Glucose` and `BMI`.
      - Generated a **correlation heatmap** to understand feature relationships.

### Phase 2: Data Preparation

1.  **Outlier Handling**:

      - Removed rows containing outliers (where any feature had a Z-score greater than 3) to reduce noise and improve model stability.

2.  **Train/Test Splitting**:

      - Split the dataset into **training (80%)** and **testing (20%)** sets to prepare for model evaluation.

3.  **Standardization**:

      - Applied **StandardScaler** to the features to normalize their range, ensuring that models like SVM and Logistic Regression perform optimally.

### Phase 3: Model Building & Training

1.  **Models Trained**:

      - **Logistic Regression**: A linear model used as a baseline.
      - **Support Vector Machine (SVM)**: A powerful classifier effective in high-dimensional spaces.
      - **Random Forest**: An ensemble model known for its high accuracy and robustness.

2.  **Hyperparameter Tuning**:

      - Used **GridSearchCV** with 5-fold cross-validation to find the optimal hyperparameters for each model, using `roc_auc` as the scoring metric.

3.  **Evaluation**:

      - Assessed models on the test set using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
      - **Random Forest** emerged as the best-performing model on the test set, with an accuracy of **75.4%** and an ROC-AUC of **0.743**.
      - **Logistic Regression** showed the best generalization with the highest cross-validated ROC-AUC score of **0.845**.

### Phase 4: Model Prediction Engine

  - Built a `predict_diabetes` function using the best model (**Random Forest**).
  - The function takes a dictionary of patient data as input.
  - It returns a prediction ("Diabetic" or "Non-Diabetic") along with a confidence score.
  - Tested the prediction engine on five distinct patient profiles to demonstrate its practical application.

-----

## Technologies Used

  - Python
  - Pandas, NumPy
  - Matplotlib, Seaborn, Missingno (for EDA/visualization)
  - Scikit-learn (for modeling, preprocessing, and evaluation)

-----

## How to Use

1.  Clone this repository:
    ```bash
    git clone https://github.com/YoussefG02/gtc-ml-project2-diabetes.git
    ```
