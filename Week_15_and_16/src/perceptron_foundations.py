"""
Perceptron Foundations & Activation Functions
Week 15 & 16: Deep Learning Foundations

This module covers:
1. The classic Rosenblatt Perceptron implemented from scratch in pure NumPy.
2. Demonstration on Boolean logic gates: AND, OR, and the famous XOR limitation (Linear Separability).
3. Core Neural Network Activation Functions and their analytical derivatives:
   - Sigmoid
   - Rectified Linear Unit (ReLU)
   - Leaky ReLU
   - Hyperbolic Tangent (Tanh)
   - Softmax (numerically stable)
4. Theoretical walkthrough of Forward & Backward propagation with Chain Rule.
5. Automated generation of publication-quality plots comparing activations & gradients.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# 1. THE CLASSICAL PERCEPTRON (Rosenblatt, 1958)
# =====================================================================

class Perceptron:
    """
    Single-layer binary perceptron classifier using Heaviside step activation.
    Linear boundary: z = w1*x1 + w2*x2 + ... + b
    Prediction: 1 if z >= 0 else 0
    Weight update rule: w = w + eta * (y - y_hat) * x
    """
    def __init__(self, learning_rate: float = 0.1, max_epochs: int = 100):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        self.epoch_errors = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.epoch_errors = []

        for epoch in range(self.max_epochs):
            total_errors = 0
            for xi, target in zip(X, y):
                linear_output = np.dot(xi, self.weights) + self.bias
                prediction = 1 if linear_output >= 0.0 else 0
                error = target - prediction
                if error != 0:
                    update = self.learning_rate * error
                    self.weights += update * xi
                    self.bias += update
                    total_errors += 1
            self.epoch_errors.append(total_errors)
            if total_errors == 0:
                # Fully converged
                break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0.0, 1, 0)


# =====================================================================
# 2. ACTIVATION FUNCTIONS & ANALYTICAL DERIVATIVES
# =====================================================================

class Activations:
    """
    Mathematical implementations of activation functions and their exact gradients.
    """

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid: sigma(z) = 1 / (1 + exp(-z))
        Range: (0, 1)
        """
        # Numerically stable clipping to prevent overflow
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    @staticmethod
    def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
        """
        d/dz sigma(z) = sigma(z) * (1 - sigma(z))
        Max gradient is 0.25 at z=0 (Root cause of Vanishing Gradient problem).
        """
        s = Activations.sigmoid(z)
        return s * (1.0 - s)

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """
        Rectified Linear Unit: max(0, z)
        Range: [0, inf)
        """
        return np.maximum(0.0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """
        d/dz ReLU(z) = 1 if z > 0 else 0
        Solves vanishing gradient for positive values; susceptible to 'Dying ReLU'.
        """
        return np.where(z > 0.0, 1.0, 0.0)

    @staticmethod
    def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """
        Leaky ReLU: z if z > 0 else alpha * z
        Range: (-inf, inf)
        Prevents Dying ReLU by retaining a small non-zero slope for z <= 0.
        """
        return np.where(z > 0.0, z, alpha * z)

    @staticmethod
    def leaky_relu_derivative(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """
        d/dz Leaky_ReLU(z) = 1 if z > 0 else alpha
        """
        return np.where(z > 0.0, 1.0, alpha)

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        """
        Hyperbolic Tangent: tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
        Range: (-1, 1), zero-centered.
        """
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z: np.ndarray) -> np.ndarray:
        """
        d/dz tanh(z) = 1 - tanh^2(z)
        Max gradient is 1.0 at z=0.
        """
        return 1.0 - np.tanh(z) ** 2

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable Softmax for multiclass output.
        softmax(z)_i = exp(z_i - max(z)) / sum(exp(z_j - max(z)))
        Range: (0, 1), sums to 1.
        """
        if z.ndim == 1:
            shifted = z - np.max(z)
            exp_z = np.exp(shifted)
            return exp_z / np.sum(exp_z)
        else:
            # Batch dimension along axis=1
            shifted = z - np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(shifted)
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# =====================================================================
# 3. VISUALIZATION OF ACTIVATIONS AND DERIVATIVES
# =====================================================================

def plot_activation_functions(output_path: str = "outputs/activation_functions.png"):
    """Plots all 4 continuous activation functions alongside their analytical derivatives."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    z = np.linspace(-5, 5, 500)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Neural Network Activation Functions & Their Derivatives", fontsize=16, fontweight="bold")

    configs = [
        ("Sigmoid", Activations.sigmoid(z), Activations.sigmoid_derivative(z), "blue", "navy", axes[0, 0]),
        ("Tanh (Hyperbolic Tangent)", Activations.tanh(z), Activations.tanh_derivative(z), "red", "darkred", axes[0, 1]),
        ("ReLU", Activations.relu(z), Activations.relu_derivative(z), "green", "darkgreen", axes[1, 0]),
        ("Leaky ReLU (alpha=0.1)", Activations.leaky_relu(z, 0.1), Activations.leaky_relu_derivative(z, 0.1), "purple", "indigo", axes[1, 1])
    ]

    for title, act_vals, grad_vals, c1, c2, ax in configs:
        ax.plot(z, act_vals, label="f(z) [Activation]", color=c1, linewidth=2.5)
        ax.plot(z, grad_vals, label="f '(z) [Derivative]", color=c2, linestyle="--", linewidth=2.0)
        ax.axhline(0, color="black", linestyle=":", alpha=0.6)
        ax.axvline(0, color="black", linestyle=":", alpha=0.6)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("z (Input)", fontsize=11)
        ax.set_ylabel("Output / Gradient", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved activation functions plot to: {output_path}")


# =====================================================================
# 4. LOGIC GATES EXPERIMENT (AND / OR / XOR)
# =====================================================================

def test_logic_gates():
    """
    Tests Rosenblatt's Perceptron on Boolean logic gates.
    Demonstrates convergence on AND and OR, and proves the XOR failure.
    """
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    gates = {
        "AND Gate (Linearly Separable)": np.array([0, 0, 0, 1]),
        "OR Gate  (Linearly Separable)": np.array([0, 1, 1, 1]),
        "XOR Gate (Non-Linearly Separable)": np.array([0, 1, 1, 0]),
    }

    print("\n" + "=" * 70)
    print(" PERCEPTRON LOGIC GATE EXPERIMENTS (Minsky & Papert 1969 Demonstration)")
    print("=" * 70)

    for gate_name, y in gates.items():
        p = Perceptron(learning_rate=0.1, max_epochs=20)
        p.fit(X, y)
        preds = p.predict(X)
        converged = np.array_equal(preds, y)

        print(f"\nEvaluating: {gate_name}")
        print(f"Inputs X:\n{X}")
        print(f"True Labels y:       {y.tolist()}")
        print(f"Predicted Labels:    {preds.tolist()}")
        print(f"Converged:           {'YES (100% Accuracy)' if converged else 'FAILED (Linear Inseparability)'}")
        print(f"Learned Weights:     {np.round(p.weights, 3)} | Bias: {round(p.bias, 3)}")
        print(f"Epoch Error History: {p.epoch_errors}")


if __name__ == "__main__":
    test_logic_gates()
    plot_activation_functions()
