"""
Loss Functions & Optimization Algorithms (SGD, Momentum, Adam)
Week 15 & 16: Deep Learning Foundations

This module covers:
1. Core Loss Functions in Machine Learning:
   - Mean Squared Error (MSE) & gradient
   - Binary Cross-Entropy (BCE) & gradient
   - Categorical Cross-Entropy (CCE) & combined Softmax-CCE gradient
2. Pure NumPy implementations of Optimizers:
   - Vanilla Stochastic Gradient Descent (SGD)
   - SGD with Momentum (Polyak, 1964)
   - Adam: Adaptive Moment Estimation (Kingma & Ba, 2014)
3. 2D Loss Surface Optimization Benchmark:
   - Comparing trajectory and convergence of SGD, Momentum, and Adam on an elongated ravine landscape.
   - Generates trajectory visualization saved to outputs/optimizer_comparison.png.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# 1. LOSS FUNCTIONS
# =====================================================================

class Losses:
    """
    Standard loss functions with numerical stability protections (epsilon clipping)
    and their exact analytical gradients.
    """

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error: (1 / (2 * m)) * sum((y_true - y_pred)^2)"""
        m = y_true.shape[0]
        return float(np.sum((y_pred - y_true) ** 2) / (2.0 * m))

    @staticmethod
    def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """d/d(y_pred) MSE = (y_pred - y_true) / m"""
        m = y_true.shape[0]
        return (y_pred - y_true) / m

    @staticmethod
    def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
        """
        BCE = - (1 / m) * sum(y * log(y_hat) + (1 - y) * log(1 - y_hat))
        """
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        loss = -np.sum(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)) / m
        return float(loss)

    @staticmethod
    def binary_cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> np.ndarray:
        """
        d/d(y_pred) BCE = (y_pred - y_true) / (y_pred * (1 - y_pred) * m)
        Note: When combined with Sigmoid output, this simplifies to (y_pred - y_true) / m.
        """
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return ((y_pred - y_true) / (y_pred * (1.0 - y_pred))) / m

    @staticmethod
    def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
        """
        CCE = - (1 / m) * sum_{i} sum_{k} y_{ik} * log(y_hat_{ik})
        y_true can be one-hot encoded or integer class indices.
        """
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        if y_true.ndim == 1:
            # Integer labels
            correct_log_probs = -np.log(y_pred[np.arange(m), y_true])
            return float(np.sum(correct_log_probs) / m)
        else:
            # One-hot encoded
            return float(-np.sum(y_true * np.log(y_pred)) / m)

    @staticmethod
    def softmax_cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Exact combined gradient of Softmax + Categorical Cross-Entropy:
        dL / dz = (y_pred - y_true) / m
        """
        m = y_true.shape[0]
        grad = y_pred.copy()
        if y_true.ndim == 1:
            grad[np.arange(m), y_true] -= 1.0
        else:
            grad -= y_true
        return grad / m


# =====================================================================
# 2. OPTIMIZERS (NUMPY)
# =====================================================================

class SGDOptimizer:
    """
    Standard Stochastic Gradient Descent (SGD) with optional Momentum.
    Momentum update:
        v = beta * v + lr * grad
        theta = theta - v
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def update(self, param_id: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if param_id not in self.velocities:
            self.velocities[param_id] = np.zeros_like(param)

        v = self.velocities[param_id]
        if self.momentum > 0.0:
            v = self.momentum * v + self.lr * grad
            self.velocities[param_id] = v
            param -= v
        else:
            param -= self.lr * grad
        return param


class AdamOptimizer:
    """
    Adaptive Moment Estimation (Adam) optimizer (Kingma & Ba, 2014).
    Maintains exponential moving averages of:
    - 1st moment (mean of gradients): m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    - 2nd moment (uncentered variance): v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t ** 2)
    With bias corrections:
        m_hat = m_t / (1 - beta1 ** t)
        v_hat = v_t / (1 - beta2 ** t)
        param = param - (lr / (sqrt(v_hat) + eps)) * m_hat
    """
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = {}

    def update(self, param_id: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(param)
            self.v[param_id] = np.zeros_like(param)
            self.t[param_id] = 0

        self.t[param_id] += 1
        t = self.t[param_id]

        m = self.m[param_id]
        v = self.v[param_id]

        # Update biased 1st and 2nd moment estimates
        m = self.beta1 * m + (1.0 - self.beta1) * grad
        v = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)

        self.m[param_id] = m
        self.v[param_id] = v

        # Bias correction
        m_hat = m / (1.0 - self.beta1 ** t)
        v_hat = v / (1.0 - self.beta2 ** t)

        # Update parameter
        param -= (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat
        return param


# =====================================================================
# 3. 2D OPTIMIZER LANDSCAPE EXPERIMENT
# =====================================================================

def loss_surface_ravine(x, y):
    """
    Elongated ravine loss surface: f(x, y) = 0.1 * x^2 + 2.0 * y^2
    The gradient along y is 20x steeper than along x, causing vanilla SGD to oscillate wildly!
    """
    return 0.1 * (x ** 2) + 2.0 * (y ** 2)

def loss_surface_gradient(x, y):
    """Analytical gradient of the ravine loss surface."""
    df_dx = 0.2 * x
    df_dy = 4.0 * y
    return np.array([df_dx, df_dy])


def run_optimizer_comparison(output_path: str = "outputs/optimizer_comparison.png"):
    """
    Simulates SGD, Momentum, and Adam starting from the same point (-4.5, 3.5)
    and tracks their trajectory towards the global minimum (0, 0).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    start_point = np.array([-4.5, 3.5])
    steps = 60

    optimizers = {
        "Vanilla SGD (lr=0.15)": (SGDOptimizer(learning_rate=0.15, momentum=0.0), "red", "o"),
        "SGD + Momentum (lr=0.15, beta=0.85)": (SGDOptimizer(learning_rate=0.15, momentum=0.85), "blue", "s"),
        "Adam (lr=0.4, beta1=0.9, beta2=0.999)": (AdamOptimizer(learning_rate=0.4), "green", "^"),
    }

    trajectories = {}

    for name, (opt, _, _) in optimizers.items():
        pos = start_point.copy()
        traj = [pos.copy()]
        for _ in range(steps):
            grad = loss_surface_gradient(pos[0], pos[1])
            pos = opt.update("pos", pos, grad)
            traj.append(pos.copy())
        trajectories[name] = np.array(traj)

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 7))

    x_range = np.linspace(-5.5, 5.5, 300)
    y_range = np.linspace(-4.0, 4.0, 300)
    X, Y = np.meshgrid(x_range, y_range)
    Z = loss_surface_ravine(X, Y)

    # Contour plot
    contours = ax.contour(X, Y, Z, levels=np.logspace(-1, 2, 25), cmap="magma", alpha=0.6)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f")

    # Plot paths
    for name, (_, color, marker) in optimizers.items():
        traj = trajectories[name]
        ax.plot(traj[:, 0], traj[:, 1], label=name, color=color, linewidth=2.0, marker=marker, markersize=4, alpha=0.9)

    ax.scatter([0], [0], color="black", s=150, marker="*", label="Global Minimum (0, 0)", zorder=10)
    ax.scatter([start_point[0]], [start_point[1]], color="black", s=80, marker="X", label="Start Point (-4.5, 3.5)", zorder=10)

    ax.set_title("Optimizer Trajectory on Ill-Conditioned Ravine: $f(x, y) = 0.1x^2 + 2y^2$", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (Slow Gradient Direction)", fontsize=11)
    ax.set_ylabel("Y (Steep Gradient Direction)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved optimizer trajectory comparison plot to: {output_path}")


if __name__ == "__main__":
    print("=" * 70)
    print(" WEEK 15 & 16: PART 3 - LOSS FUNCTIONS & OPTIMIZERS (SGD, ADAM)")
    print("=" * 70)
    run_optimizer_comparison()
