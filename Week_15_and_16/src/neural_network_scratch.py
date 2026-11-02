"""
Deep Neural Network from Scratch in Pure NumPy
Week 15 & 16: Deep Learning Foundations

Features:
- Modular Dense layer with Xavier/Glorot and He/Kaiming initialization.
- Arbitrary depth & customizable activation per layer (ReLU, Leaky ReLU, Tanh, Sigmoid, Softmax).
- Full vectorized forward propagation with activation caching.
- Vectorized backward propagation using the Chain Rule (computing dW, db, and dA_prev).
- Pluggable Optimizers: Vanilla SGD, SGD + Momentum, and Adam.
- Mini-batch gradient descent with shuffling.
- Comprehensive training metrics (Loss, Accuracy curves).
- Decision boundary plotting for non-linear synthetic data.
- High-performance execution on MNIST Handwritten Digits dataset.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.perceptron_foundations import Activations
from src.loss_and_optimizers import Losses, SGDOptimizer, AdamOptimizer


# =====================================================================
# 1. MODULAR DENSE LAYER
# =====================================================================

class DenseLayer:
    """
    Fully connected dense layer: Z = X * W + b, A = activation(Z)
    """
    def __init__(self, n_in: int, n_out: int, activation: str = "relu", init_type: str = "he", layer_id: str = "layer"):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_name = activation.lower()
        self.layer_id = layer_id

        # Weight Initialization
        if init_type == "he" or self.activation_name in ["relu", "leaky_relu"]:
            # He / Kaiming initialization (ideal for ReLU variants)
            limit = np.sqrt(2.0 / n_in)
            self.W = np.random.randn(n_in, n_out) * limit
        elif init_type == "xavier" or self.activation_name in ["tanh", "sigmoid", "softmax"]:
            # Xavier / Glorot initialization
            limit = np.sqrt(1.0 / n_in)
            self.W = np.random.randn(n_in, n_out) * limit
        else:
            self.W = np.random.randn(n_in, n_out) * 0.01

        self.b = np.zeros((1, n_out))

        # Caches for backward propagation
        self.A_prev = None
        self.Z = None
        self.A = None

        # Gradients
        self.dW = None
        self.db = None

    def forward(self, A_prev: np.ndarray) -> np.ndarray:
        """Forward pass through linear transform and non-linear activation."""
        self.A_prev = A_prev
        self.Z = np.dot(A_prev, self.W) + self.b

        if self.activation_name == "relu":
            self.A = Activations.relu(self.Z)
        elif self.activation_name == "leaky_relu":
            self.A = Activations.leaky_relu(self.Z)
        elif self.activation_name == "tanh":
            self.A = Activations.tanh(self.Z)
        elif self.activation_name == "sigmoid":
            self.A = Activations.sigmoid(self.Z)
        elif self.activation_name == "softmax":
            self.A = Activations.softmax(self.Z)
        elif self.activation_name == "linear":
            self.A = self.Z
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

        return self.A

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """
        Backpropagation for this layer:
        Given dZ = dL / dZ:
          dW = (1/m) * (A_prev^T . dZ)
          db = (1/m) * sum(dZ, axis=0, keepdims=True)
          dA_prev = dZ . W^T
        """
        m = self.A_prev.shape[0]
        self.dW = np.dot(self.A_prev.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)
        dA_prev = np.dot(dZ, self.W.T)
        return dA_prev


# =====================================================================
# 2. NEURAL NETWORK CLASS
# =====================================================================

class NeuralNetwork:
    """
    Arbitrary-depth Multi-Layer Perceptron (MLP) built from scratch in NumPy.
    """
    def __init__(self):
        self.layers = []
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def add(self, layer: DenseLayer):
        """Adds a DenseLayer to the sequential network."""
        layer.layer_id = f"layer_{len(self.layers) + 1}"
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Passes input forward through all layers."""
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, y_true: np.ndarray, loss_type: str = "categorical_cross_entropy"):
        """
        Performs full backward pass through all layers using Chain Rule.
        Computes analytical gradients dW and db for each layer.
        """
        m = y_true.shape[0]
        output_layer = self.layers[-1]
        A_out = output_layer.A

        # 1. Output Layer Gradient (dZ_L)
        if loss_type == "categorical_cross_entropy":
            # Softmax + Categorical Cross-Entropy derivative: (y_hat - y)
            dZ = Losses.softmax_cross_entropy_derivative(y_true, A_out)
        elif loss_type == "binary_cross_entropy":
            # Sigmoid + Binary Cross-Entropy derivative: (y_hat - y) / m
            if y_true.ndim == 1:
                y_true = y_true.reshape(-1, 1)
            dZ = (A_out - y_true) / m
        elif loss_type == "mse":
            dZ = Losses.mse_derivative(y_true, A_out)
            if output_layer.activation_name == "sigmoid":
                dZ = dZ * Activations.sigmoid_derivative(output_layer.Z)
            elif output_layer.activation_name == "tanh":
                dZ = dZ * Activations.tanh_derivative(output_layer.Z)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # 2. Propagate backward through preceding layers
        dA_prev = output_layer.backward(dZ)

        for layer in reversed(self.layers[:-1]):
            # dZ = dA * activation'(Z)
            if layer.activation_name == "relu":
                dZ = dA_prev * Activations.relu_derivative(layer.Z)
            elif layer.activation_name == "leaky_relu":
                dZ = dA_prev * Activations.leaky_relu_derivative(layer.Z)
            elif layer.activation_name == "tanh":
                dZ = dA_prev * Activations.tanh_derivative(layer.Z)
            elif layer.activation_name == "sigmoid":
                dZ = dA_prev * Activations.sigmoid_derivative(layer.Z)
            else:
                dZ = dA_prev

            dA_prev = layer.backward(dZ)

    def step(self, optimizer):
        """Updates parameters for all layers using the specified optimizer."""
        for layer in self.layers:
            layer.W = optimizer.update(f"{layer.layer_id}_W", layer.W, layer.dW)
            layer.b = optimizer.update(f"{layer.layer_id}_b", layer.b, layer.db)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, loss_type: str = "categorical_cross_entropy") -> float:
        if loss_type == "categorical_cross_entropy":
            return Losses.categorical_cross_entropy(y_true, y_pred)
        elif loss_type == "binary_cross_entropy":
            if y_true.ndim == 1:
                y_true = y_true.reshape(-1, 1)
            return Losses.binary_cross_entropy(y_true, y_pred)
        elif loss_type == "mse":
            return Losses.mse(y_true, y_pred)
        return 0.0

    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, loss_type: str = "categorical_cross_entropy") -> float:
        if loss_type == "categorical_cross_entropy":
            if y_true.ndim > 1:
                y_true_labels = np.argmax(y_true, axis=1)
            else:
                y_true_labels = y_true
            y_pred_labels = np.argmax(y_pred, axis=1)
            return float(np.mean(y_true_labels == y_pred_labels))
        else:
            # Binary classification
            y_pred_bin = (y_pred >= 0.5).astype(int).ravel()
            return float(np.mean(y_true.ravel() == y_pred_bin))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            epochs: int = 20, batch_size: int = 64,
            optimizer = None, loss_type: str = "categorical_cross_entropy",
            validation_data = None, verbose: bool = True):
        """
        Trains the neural network using mini-batch gradient descent.
        """
        if optimizer is None:
            optimizer = AdamOptimizer(learning_rate=0.001)

        n_samples = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            # Shuffle dataset at each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward pass
                self.forward(X_batch)
                # Backward pass
                self.backward(y_batch, loss_type=loss_type)
                # Parameter update
                self.step(optimizer)

            # Evaluate epoch
            train_preds = self.forward(X_train)
            train_loss = self.compute_loss(y_train, train_preds, loss_type)
            train_acc = self.compute_accuracy(y_train, train_preds, loss_type)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            val_str = ""
            if validation_data is not None:
                X_val, y_val = validation_data
                val_preds = self.forward(X_val)
                val_loss = self.compute_loss(y_val, val_preds, loss_type)
                val_acc = self.compute_accuracy(y_val, val_preds, loss_type)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                val_str = f" | Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1 or epoch == epochs):
                print(f"Epoch [{epoch:03d}/{epochs:03d}] -> Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}{val_str}")

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.forward(X)
        if self.layers[-1].activation_name == "softmax":
            return np.argmax(preds, axis=1)
        elif self.layers[-1].activation_name == "sigmoid":
            return (preds >= 0.5).astype(int).ravel()
        return preds


# =====================================================================
# 3. DECISION BOUNDARY PLOTTING UTILITY
# =====================================================================

def plot_decision_boundary(model: NeuralNetwork, X: np.ndarray, y: np.ndarray, title: str, output_path: str):
    """Visualizes non-linear classification boundary learned by the neural network."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]

    probs = model.forward(grid)
    if probs.shape[1] > 1:
        preds = np.argmax(probs, axis=1)
    else:
        preds = (probs >= 0.5).astype(int).ravel()
    preds = preds.reshape(xx.shape)

    plt.figure(figsize=(9, 7))
    plt.contourf(xx, yy, preds, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=45, alpha=0.9)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Feature 1", fontsize=11)
    plt.ylabel("Feature 2", fontsize=11)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved decision boundary plot to: {output_path}")
