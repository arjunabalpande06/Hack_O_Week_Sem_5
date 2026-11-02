# PBL Hack-O-Week: Week 15 & Week 16
## Machine Learning Model Deployment & Deep Learning Foundations from Scratch

This repository contains a complete, rigorous implementation of:
1. **Model Deployment Basics**: Training an end-to-end classification pipeline, comparing serialization using `pickle` vs. `joblib`, and building an inference service.
2. **Perceptron Foundations & Activations**: Classical Rosenblatt Perceptron, logic gates (AND, OR, XOR linear separability limits), and analytical activation functions with derivatives.
3. **Loss Functions & Optimizers**: Analytical gradients for MSE, Binary Cross-Entropy, and Softmax + Categorical Cross-Entropy; custom implementations of Vanilla SGD, Momentum, and Adam; 2D ravine loss landscape visualization.
4. **Deep Neural Network from Scratch in Pure NumPy**: Vectorized forward/backward propagation via the Chain Rule, mini-batch training, and benchmarks on synthetic non-linear 2D data and the **MNIST Handwritten Digits** dataset.

---

## 📁 Repository Structure

```text
Week 15 & Week 16/
├── data/
│   ├── heart.csv                      # Clinical tabular dataset for deployment
│   └── mnist_train.csv                # 784-pixel image dataset for neural network benchmark
├── models/
│   ├── heart_pipeline_pickle.pkl      # Serialized with standard Python pickle
│   ├── heart_pipeline_joblib.joblib   # Serialized with Joblib (optimized for NumPy arrays)
│   └── heart_pipeline_compressed.joblib # Serialized with Joblib (zlib compression)
├── outputs/
│   ├── activation_functions.png       # Plots of Sigmoid, Tanh, ReLU, Leaky ReLU & derivatives
│   ├── optimizer_comparison.png       # Trajectory comparison on 2D ill-conditioned ravine
│   ├── moons_decision_boundary.png    # Non-linear classification boundary on Two Moons
│   ├── mnist_sgd_vs_adam.png          # Loss and Accuracy curves comparing SGD vs. Adam
│   └── mnist_sample_predictions.png   # Sample test digit predictions with confidence scores
├── src/
│   ├── deployment_basics.py           # Model training, serialization, benchmarking, inference
│   ├── perceptron_foundations.py      # Perceptron logic gates, activation functions & derivatives
│   ├── loss_and_optimizers.py         # Loss functions, SGD, Momentum, Adam optimizers
│   └── neural_network_scratch.py      # Vectorized multi-layer neural network in pure NumPy
├── run_deployment_demo.py             # Entry script for model deployment
├── run_nn_experiments.py              # Entry script for all neural network experiments
└── README.md                          # Project documentation and theoretical report
```

---

## 🔬 Theoretical Foundations

### 1. Model Deployment: Pickle vs. Joblib
* **`pickle`**: Python’s default object serialization library. Works recursively on general Python object graphs.
* **`joblib`**: Specifically engineered for scientific computing. Internally handles large NumPy memory buffers efficiently and supports transparent compression.
* **Key Findings**:
  * Joblib with compression (`compress=3`) dramatically reduces model size on disk (e.g. from ~640 KB down to ~155 KB) while maintaining bit-exact prediction fidelity.
  * In production systems, deploying an entire `Pipeline` (e.g., `StandardScaler` + `RandomForestClassifier`) eliminates data leakage and ensures incoming raw dictionaries are preprocessed identically to training data.

### 2. The Perceptron & The XOR Dilemma
* The Rosenblatt Perceptron calculates:
  $$\hat{y} = \begin{cases} 1 & \text{if } \mathbf{w}^T \mathbf{x} + b \ge 0 \\ 0 & \text{otherwise} \end{cases}$$
* **Linear Separability**: The Perceptron can easily separate AND and OR logic gates because their positive and negative classes can be separated by a single hyperplane.
* **The XOR Failure**: Minsky and Papert (1969) proved that XOR requires a non-linear decision boundary. A single perceptron oscillates endlessly and cannot converge. Adding a hidden layer with non-linear activation functions resolves this limitation.

### 3. Activation Functions & Gradients
| Activation | Equation $f(z)$ | Derivative $f'(z)$ | Range | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$ | $\sigma(z)(1 - \sigma(z))$ | $(0, 1)$ | Saturates; max gradient $0.25$ causes vanishing gradients. |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z)$ | $(-1, 1)$ | Zero-centered, max gradient $1.0$. |
| **ReLU** | $\max(0, z)$ | $1 \text{ if } z > 0 \text{ else } 0$ | $[0, \infty)$ | Constant gradient $1$ for $z>0$; avoids vanishing gradient. |
| **Leaky ReLU** | $\max(\alpha z, z)$ | $1 \text{ if } z > 0 \text{ else } \alpha$ | $(-\infty, \infty)$ | Solves the "Dying ReLU" problem for negative activations. |
| **Softmax** | $\frac{e^{z_i - \max(z)}}{\sum e^{z_j}}$ | Combined with CCE: $\hat{y} - y$ | $(0, 1)$ | Multi-class probability distribution summing to $1.0$. |

### 4. Loss Functions & Optimizers
* **Categorical Cross-Entropy with Softmax**:
  $$L = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y_{ik} \ln(\hat{y}_{ik})$$
  The combined backward gradient reduces to the elegant difference:
  $$\frac{\partial L}{\partial \mathbf{z}^{[L]}} = \frac{1}{m} (\mathbf{\hat{y}} - \mathbf{y})$$
* **Adam Optimizer (Kingma & Ba, 2014)**:
  Combines the advantages of Momentum (first moment $m_t$) and RMSProp (second uncentered moment $v_t$):
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
  $$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 🚀 How to Run

### Step 1: Run Model Deployment Demo
```bash
python run_deployment_demo.py
```
* Trains the pipeline on `data/heart.csv`.
* Saves artifacts to `models/`.
* Benchmarks serialization and demonstrates inference on simulated patient records.

### Step 2: Run Neural Network & Deep Learning Suite
```bash
python run_nn_experiments.py
```
* Evaluates Rosenblatt Perceptron on logic gates.
* Trains 2-layer Neural Net solving XOR with 100% accuracy.
* Computes and plots non-linear decision boundary on Two Moons.
* Trains multi-layer neural network on MNIST (SGD vs. Adam comparison).
* Saves all visualization artifacts to `outputs/`.
