# 🎬 Simple Movie Recommendation System Using Linear Algebra & Machine Learning

An interactive, full-stack Python and Streamlit web application that implements a movie recommendation engine from first principles. Designed to demonstrate all **7 core Linear Algebra and Calculus topics**:
1. **Vectors**: Movie representations as feature vectors in genre space
2. **Matrices**: Movie-Feature Matrix & Batch operations
3. **Dot Product**: Similarity computation (Dot Product & Cosine Similarity)
4. **Eigenvalues & Eigenvectors**: Principal Component Analysis (PCA) for 2D visualization
5. **Derivatives**: Rating prediction error loss formulation
6. **Gradients**: Multivariable preference gradient vectors
7. **Chain Rule**: Gradient Descent optimization / backpropagation

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will launch automatically in your web browser at `http://localhost:8501`.

---

## 📂 Project Architecture

```
├── app.py                     # Streamlit application with multi-tab interface & custom styling
├── dataset.py                 # 45+ movies dataset with 8 genre features
├── recommender.py             # Vector dot product & cosine similarity recommendation logic
├── linear_algebra.py          # Covariance matrix, Eigenvalues, Eigenvectors, & PCA implementation
├── ml_engine.py               # Gradient descent user preference learning model
├── visualizer.py              # Interactive Plotly charts (2D PCA plot, loss curve, heatmaps)
├── requirements.txt           # Required Python packages
└── README.md                  # Comprehensive project documentation
```

---

## 📚 7 Syllabus Topics Explained

| # | Topic | Implementation in Project | Mathematical Formula |
|---|---|---|---|
| 1 | **Vectors** | Represents movies as 8D genre weight vectors ($v_i$) | $v = [v_{\text{action}}, v_{\text{comedy}}, \dots, v_{\text{fantasy}}]^T$ |
| 2 | **Matrices** | Stored in Movie-Feature Matrix $X \in \mathbb{R}^{N \times 8}$ | $S = X_{\text{norm}} X_{\text{norm}}^T$ |
| 3 | **Dot Product** | Calculates movie similarity | $u \cdot v = \sum_{i=1}^d u_i v_i$ |
| 4 | **Eigenvalues & Eigenvectors** | Eigen-decomposition of Covariance Matrix for PCA | $C e_k = \lambda_k e_k$ |
| 5 | **Derivatives** | Rating prediction error derivative | $\frac{\partial L}{\partial \hat{y}} = \hat{y} - y = e$ |
| 6 | **Gradients** | Multivariable gradient vector of loss | $\nabla_w L = e \mathbf{v}_i$ |
| 7 | **Chain Rule** | Backpropagation to update user preferences | $\frac{\partial L}{\partial w_j} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_j} = e \cdot v_j$ |

---

## 🧪 Verification & Unit Testing

You can verify all modules by running Python compilation checks:

```bash
python -m py_compile app.py dataset.py recommender.py linear_algebra.py ml_engine.py visualizer.py
```

Enjoy building and learning! 🎬
