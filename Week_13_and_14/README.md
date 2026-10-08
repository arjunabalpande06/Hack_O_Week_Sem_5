# 📊 Telco Customer Churn: Ensemble Learning, Regularization & Bias-Variance Diagnostics

**Academic Practical / Hack-O-Week Project (Weeks 13 & 14)**  
**Domain**: Applied Machine Learning, Model Diagnostics, and Predictive Analytics  
**Primary Artifact**: [`notebooks/ensemble_bias_variance_churn.ipynb`](file:///d:/Symbiosis/SEM%205/PBL%20Hack%20O%20Week/Week%2013%20&%20Week%2014/notebooks/ensemble_bias_variance_churn.ipynb)

---

## 📌 Executive Summary

This project implements an end-to-end Machine Learning research workflow on the **Telco Customer Churn** dataset (~7,043 customer accounts), explicitly demonstrating and connecting:
1. **Bias-Variance Tradeoff** (Mathematical grounding & empirical validation)
2. **Underfitting vs. Overfitting** (Diagnostic signatures via learning curves and generalization gap)
3. **Regularization Techniques** (L1 Lasso, L2 Ridge, tree-pruning, boosting penalties)
4. **Ensemble Methods — Bagging** (`BaggingClassifier`, `RandomForestClassifier` with random subspace feature sampling)
5. **Ensemble Methods — Boosting** (`XGBoost`, `LightGBM` with 2nd-order Taylor expansions, histogram binning, and early stopping)

---

## 📐 Theoretical Framework

### 1. The Bias-Variance Decomposition
For an unknown relationship $y = f(x) + \epsilon$ with noise variance $\sigma^2$, the expected generalization error of an estimator $\hat{f}(x)$ is decomposed as:

$$\mathbb{E}\left[(y - \hat{f}(x))^2\right] = \underbrace{\left(\mathbb{E}[\hat{f}(x)] - f(x)\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Noise}}$$

* **High Bias (Underfitting)**: Overly simplistic assumptions prevent the model from capturing true underlying patterns (e.g., depth-1 Decision Stump). Both training and test errors are unacceptably high.
* **High Variance (Overfitting)**: Excess model capacity allows memorization of sample noise (e.g., unconstrained Decision Tree). The training error approaches zero, but test error skyrockets.
* **Generalization Gap**: Defined as $\Delta = \text{Score}_{\text{Train}} - \text{Score}_{\text{Test}}$. A large $\Delta$ is the primary empirical indicator of overfitting.

---

### 2. How Bagging & Boosting Combat Errors

```
                                  [High Error Baseline]
                                            │
                ┌───────────────────────────┴───────────────────────────┐
                ▼                                                       ▼
      [High Variance Issue]                                   [High Bias Issue]
                │                                                       │
        Ensemble: BAGGING                                       Ensemble: BOOSTING
      (Parallel Bootstrap)                                     (Sequential Residuals)
                │                                                       │
  Var = ρ·σ² + ((1-ρ)/B)·σ²                             F_m(x) = F_{m-1}(x) + η·h_m(x)
                │                                                       │
 ┌──────────────┴──────────────┐                         ┌──────────────┴──────────────┐
 ▼                             ▼                         ▼                             ▼
BaggingClassifier        Random Forest                 XGBoost                     LightGBM
(Bootstrap Aggregation)  (Subspace √p Sampling)  (2nd-order Taylor + L1/L2)   (Leaf-wise + Histogram)
```

1. **Bagging (Bootstrap Aggregation)**:
   - Trains $B$ high-variance base estimators in parallel on bootstrap samples.
   - For uncorrelated trees: $\text{Var} = \frac{\sigma^2}{B}$.
   - **Random Forest** decorrelates trees by randomly sampling $m = \sqrt{p}$ features at every split, driving down the pairwise correlation $\rho$.

2. **Boosting (Gradient Boosting / XGBoost / LightGBM)**:
   - Sequentially trains weak learners to fit the pseudo-residuals of the previous ensemble.
   - Systematically shrinks **Bias**.
   - Variance is strictly controlled via shrinkage learning rate ($\eta$), tree-depth constraints, L1/L2 regularization (`reg_alpha`, `reg_lambda`), and **Early Stopping**.

---

## 🏆 Key Benchmark Results

All models were evaluated on an identical stratified 80/20 train/test split of the preprocessed Telco dataset:

| Model Architecture | Paradigm | Train ROC-AUC | Test ROC-AUC | Test Accuracy | Test F1 | Generalization Gap (AUC) | Diagnostic Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Decision Stump (depth=1)** | Base Tree | ~0.647 | ~0.650 | ~0.735 | ~0.000 | -0.003 | ⚠️ **Severe Underfitting (High Bias)** |
| **Unpruned Decision Tree** | Base Tree | **~1.000** | ~0.658 | ~0.725 | ~0.490 | **+0.342** | 🚨 **Severe Overfitting (High Variance)** |
| **Regularized Tree (depth=4)** | Pruned Tree | ~0.835 | ~0.825 | ~0.792 | ~0.530 | +0.010 | ✅ Balanced Bias-Variance |
| **Logistic Regression (L2)** | Linear Baseline | ~0.849 | ~0.843 | ~0.803 | ~0.590 | +0.006 | ✅ Strong Linear Baseline |
| **Bagging Classifier** | Ensemble (Bagging) | ~0.999 | ~0.816 | ~0.776 | ~0.510 | +0.183 | ⚡ Significant Variance Drop over single tree |
| **Random Forest** | Ensemble (Bagging+Subspace) | ~1.000 | ~0.831 | ~0.791 | ~0.550 | +0.169 | ⚡ Enhanced Decorrelation |
| **XGBoost (Regularized)** | Ensemble (Boosting) | ~0.865 | **~0.846** | **~0.806** | **~0.601** | **+0.019** | 🌟 **Top Performer (High AUC & Tight Gap)** |
| **LightGBM (Leaf-wise)** | Ensemble (Boosting) | ~0.871 | **~0.845** | **~0.804** | **~0.598** | **+0.026** | 🌟 **Top Performer (Fastest Convergence)** |

---

## 🔬 Visualizations Generated in Notebook

The pre-executed notebook [`notebooks/ensemble_bias_variance_churn.ipynb`](file:///d:/Symbiosis/SEM%205/PBL Hack O Week/Week 13 & Week 14/notebooks/ensemble_bias_variance_churn.ipynb) contains fully generated figures:
1. **Validation Curve (`max_depth` from 1 to 20)**: Highlights underfitting at depth 1, the optimal tradeoff at depth 4, and widening divergence as depth increases.
2. **Learning Curve**: Tracks training score vs. cross-validation score across sample size.
3. **L1 vs. L2 Regularization Path**: Demonstrates how L1 Lasso forces non-predictive weights to zero (exact sparsity) vs. L2 Ridge which smoothly shrinks weights.
4. **Estimator Growth Curves**: Shows test ROC-AUC climbing and variance stabilizing as the number of trees ($N_{\text{estimators}}$) increases in Bagging and Random Forest.
5. **Early Stopping Loss Trajectory**: Visualizes XGBoost train and validation log-loss over 300 boosting rounds, demonstrating the exact point early stopping arrests training to prevent overfitting.
6. **Feature Importance (XGBoost vs. LightGBM)**: Highlights the primary churn drivers: `Contract_Two year`, `tenure`, `InternetService_Fiber optic`, and `MonthlyCharges`.
7. **Combined ROC Curves**: Overlays all model architectures on a single plot.
8. **Generalization Gap Bar Chart**: Visually flags models with large overfitting gaps ($>0.15$).

---

## 🚀 How to Open and Run

The notebook is already **100% pre-executed with all figures embedded**:
1. Open [`notebooks/ensemble_bias_variance_churn.ipynb`](file:///d:/Symbiosis/SEM%205/PBL Hack O Week/Week 13 & Week 14/notebooks/ensemble_bias_variance_churn.ipynb) in VS Code, JupyterLab, or Jupyter Notebook:
   ```powershell
   jupyter notebook "notebooks/ensemble_bias_variance_churn.ipynb"
   ```
2. To re-run or modify experiments:
   - Simply click **Run All** in your notebook interface.
   - Or run via command line:
     ```powershell
     python build_notebook.py
     jupyter nbconvert --to notebook --execute --inplace "notebooks/ensemble_bias_variance_churn.ipynb"
     ```
