import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

import xgboost as xgb
import lightgbm as lgb

# Configure aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'

os.makedirs('figures', exist_ok=True)

print("="*80)
print(" TELCO CHURN ML PIPELINE: ENSEMBLES, REGULARIZATION & BIAS-VARIANCE DIAGNOSTICS")
print("="*80)

# 1. LOAD & CLEAN DATA
print("\n[1/7] Loading and preprocessing dataset...")
data_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(data_path)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
df_clean = df.drop(columns=['customerID'])

cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True, dtype=int)

X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

print(f"Dataset successfully prepared: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows, {X.shape[1]} features.")

# 2. BIAS-VARIANCE EXPERIMENT: UNDERFIT VS OVERFIT
print("\n[2/7] Diagnosing Underfitting vs. Overfitting...")
clf_underfit = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X_train, y_train)
clf_overfit = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=42).fit(X_train, y_train)
clf_balanced = DecisionTreeClassifier(max_depth=4, min_samples_leaf=25, random_state=42).fit(X_train, y_train)

# Validation Curve Plot
print("  -> Generating Validation Curve (max_depth 1..18)...")
depths = np.arange(1, 19)
train_scores, val_scores = validation_curve(
    DecisionTreeClassifier(random_state=42),
    X_train, y_train,
    param_name="max_depth",
    param_range=depths,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
tr_mean, tr_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
val_mean, val_std = np.mean(val_scores, axis=1), np.std(val_scores, axis=1)

plt.figure(figsize=(10, 5))
plt.plot(depths, tr_mean, 'o-', color='#e74c3c', label='Train ROC-AUC', lw=2)
plt.fill_between(depths, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color='#e74c3c')
plt.plot(depths, val_mean, 's-', color='#2980b9', label='5-Fold CV ROC-AUC', lw=2)
plt.fill_between(depths, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#2980b9')
plt.axvline(x=4, color='#27ae60', linestyle='--', label='Optimal Tradeoff (depth=4)')
plt.title('Validation Curve: Bias-Variance Tradeoff across Tree Depth', fontsize=14)
plt.xlabel('Tree Maximum Depth (Complexity)', fontsize=11)
plt.ylabel('ROC-AUC Score', fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig('figures/01_validation_curve_bias_variance.png', dpi=200)
plt.close()

# 3. REGULARIZATION EXPERIMENT: L1 VS L2
print("\n[3/7] Evaluating Regularization (L1 Lasso vs. L2 Ridge)...")
c_values = [0.001, 0.01, 0.1, 1.0, 10.0]
l1_zeros, l2_zeros, l1_aucs, l2_aucs = [], [], [], []

for c in c_values:
    m_l1 = LogisticRegression(penalty='l1', C=c, solver='liblinear', random_state=42).fit(X_train_scaled, y_train)
    m_l2 = LogisticRegression(penalty='l2', C=c, solver='lbfgs', random_state=42).fit(X_train_scaled, y_train)
    l1_zeros.append(np.sum(np.isclose(m_l1.coef_, 0.0, atol=1e-4)))
    l2_zeros.append(np.sum(np.isclose(m_l2.coef_, 0.0, atol=1e-4)))
    l1_aucs.append(roc_auc_score(y_test, m_l1.predict_proba(X_test_scaled)[:, 1]))
    l2_aucs.append(roc_auc_score(y_test, m_l2.predict_proba(X_test_scaled)[:, 1]))

logreg_final = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', random_state=42).fit(X_train_scaled, y_train)

# 4. ENSEMBLE BAGGING: BAGGING CLASSIFIER & RANDOM FOREST
print("\n[4/7] Training Bagging & Random Forest models...")
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
).fit(X_train, y_train)

rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
).fit(X_train, y_train)

# 5. ENSEMBLE BOOSTING: XGBOOST & LIGHTGBM
print("\n[5/7] Training Boosting models (XGBoost & LightGBM) with Regularization & Early Stopping...")
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.5,
    early_stopping_rounds=20,
    eval_metric='logloss',
    random_state=42
).fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)

lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.5,
    random_state=42,
    verbose=-1
).fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(20, verbose=False)])

# 6. BENCHMARKING & METRICS COLLECTION
print("\n[6/7] Computing final benchmark comparison...")
all_models = {
    '1. Decision Stump (High Bias)': (clf_underfit, X_train, X_test),
    '2. Decision Tree (High Variance)': (clf_overfit, X_train, X_test),
    '3. Regularized Tree (Pruned)': (clf_balanced, X_train, X_test),
    '4. Logistic Regression (L2)': (logreg_final, X_train_scaled, X_test_scaled),
    '5. Bagging Classifier': (bagging_clf, X_train, X_test),
    '6. Random Forest (Bagging)': (rf_clf, X_train, X_test),
    '7. XGBoost (Boosting+L1/L2)': (xgb_model, X_train, X_test),
    '8. LightGBM (Boosting+Leaves)': (lgb_model, X_train, X_test)
}

records = []
for name, (mod, xtr, xte) in all_models.items():
    tr_pred = mod.predict(xtr)
    te_pred = mod.predict(xte)
    tr_prob = mod.predict_proba(xtr)[:, 1]
    te_prob = mod.predict_proba(xte)[:, 1]
    
    tr_acc = accuracy_score(y_train, tr_pred)
    te_acc = accuracy_score(y_test, te_pred)
    tr_auc = roc_auc_score(y_train, tr_prob)
    te_auc = roc_auc_score(y_test, te_prob)
    te_f1 = f1_score(y_test, te_pred)
    gap = tr_auc - te_auc
    
    records.append({
        'Model Architecture': name,
        'Train Acc': f"{tr_acc*100:.2f}%",
        'Test Acc': f"{te_acc*100:.2f}%",
        'Train AUC': f"{tr_auc:.4f}",
        'Test AUC': f"{te_auc:.4f}",
        'Test F1': f"{te_f1:.4f}",
        'Gap (Overfitting)': f"{gap:+.4f}"
    })

results_df = pd.DataFrame(records)

print("\n" + "="*95)
print(" BENCHMARK RESULTS MATRIX")
print("="*95)
print(results_df.to_string(index=False))
print("="*95)

# 7. EXPORT ROC & GENERALIZATION GAP CHARTS
print("\n[7/7] Generating publication-quality charts...")

# Chart A: ROC Curves
plt.figure(figsize=(10, 6))
for name, (mod, xtr, xte) in all_models.items():
    if 'Stump' in name:
        continue
    te_prob = mod.predict_proba(xte)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, te_prob)
    auc_v = roc_auc_score(y_test, te_prob)
    plt.plot(fpr, tpr, lw=2, label=f"{name.split('.')[1].strip()} (AUC={auc_v:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance (AUC=0.50)')
plt.title('Multi-Model ROC-AUC Comparison on Test Set', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig('figures/02_roc_curves_comparison.png', dpi=200)
plt.close()

# Chart B: Generalization Gap Bar Chart
gaps = [float(r['Gap (Overfitting)']) for r in records]
model_names = [r['Model Architecture'].split('.')[1].strip() for r in records]

plt.figure(figsize=(11, 5))
bar_colors = ['#e74c3c' if g > 0.15 else ('#f39c12' if g > 0.05 else '#27ae60') for g in gaps]
sns.barplot(x=gaps, y=model_names, palette=bar_colors)
plt.axvline(x=0.05, color='gray', linestyle='--', label='Acceptable Gap (<0.05)')
plt.title('Generalization Gap (Train AUC - Test AUC): Measuring Overfitting', fontsize=14)
plt.xlabel('Overfitting Delta (Lower is better)', fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig('figures/03_generalization_gap.png', dpi=200)
plt.close()

print("Saved figure: figures/01_validation_curve_bias_variance.png")
print("Saved figure: figures/02_roc_curves_comparison.png")
print("Saved figure: figures/03_generalization_gap.png")

print("\nPipeline execution complete! All models, diagnostics, and figures generated successfully.")
