import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Set styling for plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# 1. Load Data
df = pd.read_csv('employee_salary_dataset.csv')
print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

# Features & Preprocessing Definition
# Drop non-predictive identifiers like EmployeeID, Name
features_df = df.drop(columns=['EmployeeID', 'Name', 'Monthly_Salary'])
target_regression = df['Monthly_Salary']

# Classification target: High Salary (1 if > median, 0 otherwise)
salary_median = df['Monthly_Salary'].median()
df['High_Salary'] = (df['Monthly_Salary'] > salary_median).astype(int)
target_classification = df['High_Salary']

print(f"\nTarget Variable for Regression: Monthly_Salary (Continuous, Min: {target_regression.min()}, Max: {target_regression.max()}, Mean: {target_regression.mean():.2f})")
print(f"Target Variable for Classification: High_Salary (Binary: 0 = Low/Average <= {salary_median}, 1 = High > {salary_median})")
print(f"Class distribution: 0: {(target_classification == 0).sum()}, 1: {(target_classification == 1).sum()}")

categorical_cols = ['Department', 'Education_Level', 'Gender', 'City']
numerical_cols = ['Experience_Years', 'Age']

print(f"\nNumerical features: {numerical_cols}")
print(f"Categorical features: {categorical_cols}")

# ==============================================================================
# PART 1: REGRESSION MODELS
# ==============================================================================
print("\n" + "#"*80)
print("PART 1: REGRESSION ANALYSIS (Target: Monthly_Salary)")
print("#"*80)

# Train-Test Split for Regression (80/20 split, random_state=42 for reproducibility)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    features_df, target_regression, test_size=0.2, random_state=42
)

# Standard Preprocessor
preprocessor_linear = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ]
)

# Preprocessor for Polynomial Regression
preprocessor_poly = ColumnTransformer(
    transformers=[
        ('num_poly', Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
        ]), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ]
)

# Dictionary to store regression metrics
reg_metrics = {}
reg_predictions = {'Actual': y_test_reg.values}

# --- 1.1 Linear Regression ---
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor_linear),
    ('regressor', LinearRegression())
])
lr_pipeline.fit(X_train_reg, y_train_reg)
y_pred_lr = lr_pipeline.predict(X_test_reg)
y_train_pred_lr = lr_pipeline.predict(X_train_reg)

reg_metrics['Linear Regression'] = {
    'Train R2': r2_score(y_train_reg, y_train_pred_lr),
    'Test R2': r2_score(y_test_reg, y_pred_lr),
    'MAE': mean_absolute_error(y_test_reg, y_pred_lr),
    'MSE': mean_squared_error(y_test_reg, y_pred_lr),
    'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_lr))
}
reg_predictions['Linear'] = y_pred_lr

# --- 1.2 Polynomial Regression (Degree 2) ---
poly_pipeline = Pipeline([
    ('preprocessor', preprocessor_poly),
    ('regressor', LinearRegression())
])
poly_pipeline.fit(X_train_reg, y_train_reg)
y_pred_poly = poly_pipeline.predict(X_test_reg)
y_train_pred_poly = poly_pipeline.predict(X_train_reg)

reg_metrics['Polynomial Regression (Degree 2)'] = {
    'Train R2': r2_score(y_train_reg, y_train_pred_poly),
    'Test R2': r2_score(y_test_reg, y_pred_poly),
    'MAE': mean_absolute_error(y_test_reg, y_pred_poly),
    'MSE': mean_squared_error(y_test_reg, y_pred_poly),
    'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_poly))
}
reg_predictions['Polynomial'] = y_pred_poly

# --- 1.3 Ridge Regression (L2 Regularization) ---
ridge_pipeline = Pipeline([
    ('preprocessor', preprocessor_linear),
    ('regressor', Ridge(alpha=1.0, random_state=42))
])
ridge_pipeline.fit(X_train_reg, y_train_reg)
y_pred_ridge = ridge_pipeline.predict(X_test_reg)
y_train_pred_ridge = ridge_pipeline.predict(X_train_reg)

reg_metrics['Ridge Regression (alpha=1.0)'] = {
    'Train R2': r2_score(y_train_reg, y_train_pred_ridge),
    'Test R2': r2_score(y_test_reg, y_pred_ridge),
    'MAE': mean_absolute_error(y_test_reg, y_pred_ridge),
    'MSE': mean_squared_error(y_test_reg, y_pred_ridge),
    'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_ridge))
}
reg_predictions['Ridge'] = y_pred_ridge

# --- 1.4 Lasso Regression (L1 Regularization) ---
lasso_pipeline = Pipeline([
    ('preprocessor', preprocessor_linear),
    ('regressor', Lasso(alpha=500.0, max_iter=10000, random_state=42))
])
lasso_pipeline.fit(X_train_reg, y_train_reg)
y_pred_lasso = lasso_pipeline.predict(X_test_reg)
y_train_pred_lasso = lasso_pipeline.predict(X_train_reg)

reg_metrics['Lasso Regression (alpha=500.0)'] = {
    'Train R2': r2_score(y_train_reg, y_train_pred_lasso),
    'Test R2': r2_score(y_test_reg, y_pred_lasso),
    'MAE': mean_absolute_error(y_test_reg, y_pred_lasso),
    'MSE': mean_squared_error(y_test_reg, y_pred_lasso),
    'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_lasso))
}
reg_predictions['Lasso'] = y_pred_lasso

# Display individual regression results
df_reg_metrics = pd.DataFrame(reg_metrics).T
print("\n" + "="*80)
print("REGRESSION MODELS OVERALL COMPARISON")
print("="*80)
print(df_reg_metrics.to_string())

print("\n" + "="*80)
print("INDIVIDUAL REGRESSION MODEL RESULTS")
print("="*80)
for model_name, metrics in reg_metrics.items():
    print(f"\n>>> REGRESSION MODEL: {model_name.upper()}")
    print("-" * 60)
    print(f"  * Train R-squared (R2) : {metrics['Train R2']:.4f}")
    print(f"  * Test R-squared (R2)  : {metrics['Test R2']:.4f}")
    print(f"  * Mean Absolute Error  : {metrics['MAE']:.2f}")
    print(f"  * Mean Squared Error   : {metrics['MSE']:.2f}")
    print(f"  * Root Mean Sq Error   : {metrics['RMSE']:.2f}")

print("\n" + "="*80)
print("SAMPLE TEST PREDICTIONS VS ACTUAL SALARY")
print("="*80)
df_reg_pred = pd.DataFrame(reg_predictions)
print(df_reg_pred.head(10).round(2).to_string())


# ==============================================================================
# PART 2: CLASSIFICATION MODELS
# ==============================================================================
print("\n" + "#"*80)
print("PART 2: CLASSIFICATION ANALYSIS (Target: High_Salary [0 or 1])")
print("#"*80)

# Train-Test Split for Classification (Stratified 80/20 split)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    features_df, target_classification, test_size=0.2, random_state=42, stratify=target_classification
)

clf_metrics = {}
clf_conf_matrices = {}

# --- 2.1 Logistic Regression ---
log_reg_pipeline = Pipeline([
    ('preprocessor', preprocessor_linear),
    ('classifier', LogisticRegression(random_state=42, C=1.0))
])
log_reg_pipeline.fit(X_train_clf, y_train_clf)
y_pred_lr_clf = log_reg_pipeline.predict(X_test_clf)
y_pred_proba_lr_clf = log_reg_pipeline.predict_proba(X_test_clf)[:, 1]

clf_metrics['Logistic Regression'] = {
    'Accuracy': accuracy_score(y_test_clf, y_pred_lr_clf),
    'Precision': precision_score(y_test_clf, y_pred_lr_clf, zero_division=0),
    'Recall': recall_score(y_test_clf, y_pred_lr_clf, zero_division=0),
    'F1-Score': f1_score(y_test_clf, y_pred_lr_clf, zero_division=0),
    'ROC-AUC': roc_auc_score(y_test_clf, y_pred_proba_lr_clf)
}
clf_conf_matrices['Logistic Regression'] = confusion_matrix(y_test_clf, y_pred_lr_clf)

# --- 2.2 K-Nearest Neighbors (KNN) Classifier ---
knn_pipeline = Pipeline([
    ('preprocessor', preprocessor_linear),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])
knn_pipeline.fit(X_train_clf, y_train_clf)
y_pred_knn = knn_pipeline.predict(X_test_clf)
y_pred_proba_knn = knn_pipeline.predict_proba(X_test_clf)[:, 1]

clf_metrics['K-Nearest Neighbors (k=5)'] = {
    'Accuracy': accuracy_score(y_test_clf, y_pred_knn),
    'Precision': precision_score(y_test_clf, y_pred_knn, zero_division=0),
    'Recall': recall_score(y_test_clf, y_pred_knn, zero_division=0),
    'F1-Score': f1_score(y_test_clf, y_pred_knn, zero_division=0),
    'ROC-AUC': roc_auc_score(y_test_clf, y_pred_proba_knn)
}
clf_conf_matrices['K-Nearest Neighbors (k=5)'] = confusion_matrix(y_test_clf, y_pred_knn)

# Also test k=3 for comparison
knn3_pipeline = Pipeline([
    ('preprocessor', preprocessor_linear),
    ('classifier', KNeighborsClassifier(n_neighbors=3))
])
knn3_pipeline.fit(X_train_clf, y_train_clf)
y_pred_knn3 = knn3_pipeline.predict(X_test_clf)
y_pred_proba_knn3 = knn3_pipeline.predict_proba(X_test_clf)[:, 1]

clf_metrics['K-Nearest Neighbors (k=3)'] = {
    'Accuracy': accuracy_score(y_test_clf, y_pred_knn3),
    'Precision': precision_score(y_test_clf, y_pred_knn3, zero_division=0),
    'Recall': recall_score(y_test_clf, y_pred_knn3, zero_division=0),
    'F1-Score': f1_score(y_test_clf, y_pred_knn3, zero_division=0),
    'ROC-AUC': roc_auc_score(y_test_clf, y_pred_proba_knn3)
}
clf_conf_matrices['K-Nearest Neighbors (k=3)'] = confusion_matrix(y_test_clf, y_pred_knn3)

# Display individual classification results
df_clf_metrics = pd.DataFrame(clf_metrics).T
print("\n" + "="*80)
print("CLASSIFICATION MODELS OVERALL COMPARISON")
print("="*80)
print(df_clf_metrics.to_string())

print("\n" + "="*80)
print("INDIVIDUAL CLASSIFICATION MODEL RESULTS & DETAILED METRICS")
print("="*80)

for model_name in ['Logistic Regression', 'K-Nearest Neighbors (k=5)', 'K-Nearest Neighbors (k=3)']:
    print(f"\n>>> MODEL: {model_name.upper()}")
    print("-" * 60)
    print(f"  * Accuracy:        {clf_metrics[model_name]['Accuracy'] * 100:.2f}% ({clf_metrics[model_name]['Accuracy']:.4f})")
    print(f"  * Precision:       {clf_metrics[model_name]['Precision'] * 100:.2f}% ({clf_metrics[model_name]['Precision']:.4f})")
    print(f"  * Recall:          {clf_metrics[model_name]['Recall'] * 100:.2f}% ({clf_metrics[model_name]['Recall']:.4f})")
    print(f"  * F1-Score:        {clf_metrics[model_name]['F1-Score']:.4f}")
    print(f"  * ROC-AUC:         {clf_metrics[model_name]['ROC-AUC']:.4f}")
    print("\n  * Confusion Matrix:")
    cm = clf_conf_matrices[model_name]
    print(f"    [[True Negative (TN) ={cm[0,0]}  False Positive (FP)={cm[0,1]}]")
    print(f"     [False Negative (FN)={cm[1,0]}  True Positive (TP) ={cm[1,1]}]]")
    
    if model_name == 'Logistic Regression':
        y_pred = y_pred_lr_clf
    elif model_name == 'K-Nearest Neighbors (k=5)':
        y_pred = y_pred_knn
    else:
        y_pred = y_pred_knn3
        
    print("\n  * Detailed Classification Report:")
    print(classification_report(y_test_clf, y_pred, target_names=['Low/Avg Salary (0)', 'High Salary (1)'], digits=4))



# ==============================================================================
# PART 3: GENERATE PLOTS AND VISUALIZATIONS
# ==============================================================================
os.makedirs('visualizations', exist_ok=True)

# Figure 1: Regression Results (Actual vs Predicted & Metrics Bar Chart)
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Regression Models Evaluation on Employee Salary Dataset', fontsize=16, fontweight='bold')

# Plot 1: Linear Regression
axes[0, 0].scatter(y_test_reg, y_pred_lr, color='#2b5c8f', alpha=0.8, edgecolors='k', s=60, label='Predictions')
min_val = min(y_test_reg.min(), y_pred_lr.min()) - 5000
max_val = max(y_test_reg.max(), y_pred_lr.max()) + 5000
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit ($y=x$)')
axes[0, 0].set_title(f"Linear Regression\n$R^2$: {reg_metrics['Linear Regression']['Test R2']:.3f} | RMSE: {reg_metrics['Linear Regression']['RMSE']:.1f}", fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('Actual Salary (₹)')
axes[0, 0].set_ylabel('Predicted Salary (₹)')
axes[0, 0].legend()
axes[0, 0].grid(True, linestyle='--', alpha=0.6)

# Plot 2: Polynomial Regression
axes[0, 1].scatter(y_test_reg, y_pred_poly, color='#e056fd', alpha=0.8, edgecolors='k', s=60, label='Predictions')
axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit ($y=x$)')
axes[0, 1].set_title(f"Polynomial Regression (Deg 2)\n$R^2$: {reg_metrics['Polynomial Regression (Degree 2)']['Test R2']:.3f} | RMSE: {reg_metrics['Polynomial Regression (Degree 2)']['RMSE']:.1f}", fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Actual Salary (₹)')
axes[0, 1].set_ylabel('Predicted Salary (₹)')
axes[0, 1].legend()
axes[0, 1].grid(True, linestyle='--', alpha=0.6)

# Plot 3: Ridge Regression
axes[1, 0].scatter(y_test_reg, y_pred_ridge, color='#00b894', alpha=0.8, edgecolors='k', s=60, label='Predictions')
axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit ($y=x$)')
axes[1, 0].set_title(f"Ridge Regression (alpha=1.0)\n$R^2$: {reg_metrics['Ridge Regression (alpha=1.0)']['Test R2']:.3f} | RMSE: {reg_metrics['Ridge Regression (alpha=1.0)']['RMSE']:.1f}", fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('Actual Salary (₹)')
axes[1, 0].set_ylabel('Predicted Salary (₹)')
axes[1, 0].legend()
axes[1, 0].grid(True, linestyle='--', alpha=0.6)

# Plot 4: Lasso Regression
axes[1, 1].scatter(y_test_reg, y_pred_lasso, color='#d63031', alpha=0.8, edgecolors='k', s=60, label='Predictions')
axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit ($y=x$)')
axes[1, 1].set_title(f"Lasso Regression (alpha=500.0)\n$R^2$: {reg_metrics['Lasso Regression (alpha=500.0)']['Test R2']:.3f} | RMSE: {reg_metrics['Lasso Regression (alpha=500.0)']['RMSE']:.1f}", fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('Actual Salary (₹)')
axes[1, 1].set_ylabel('Predicted Salary (₹)')
axes[1, 1].legend()
axes[1, 1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('visualizations/regression_predictions.png')
plt.close()

# Figure 2: Classification Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
fig.suptitle('Classification Models Confusion Matrices (Target: High Salary Level)', fontsize=15, fontweight='bold')

for ax, (model_name, cm) in zip(axes, clf_conf_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Low/Avg', 'High'], yticklabels=['Low/Avg', 'High'],
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(f"{model_name}\nAcc: {clf_metrics[model_name]['Accuracy']*100:.1f}% | F1: {clf_metrics[model_name]['F1-Score']:.3f}", fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

plt.tight_layout()
plt.savefig('visualizations/classification_confusion_matrices.png')
plt.close()

print("\nVisualizations successfully saved to 'visualizations/' folder.")
