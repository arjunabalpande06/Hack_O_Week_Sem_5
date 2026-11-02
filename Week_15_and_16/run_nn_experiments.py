"""
Comprehensive Deep Learning & Neural Network Experiments (NumPy from Scratch)
Week 15 & 16: Deep Learning Foundations

Experiments executed:
1. Perceptron Logic Gates (Rosenblatt) vs. 2-Layer Neural Network solving XOR.
2. Non-linear 2D Decision Boundary on Synthetic Moon data.
3. Handwritten Digits Classification (MNIST / Digit Recognizer):
   - Training MLP from scratch with Softmax + Categorical Cross-Entropy.
   - Benchmark: Vanilla SGD vs. Adam Optimizer.
   - Visualizing Training Curves (Loss & Accuracy).
   - Visualizing real digit inferences with confidence percentages.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from src.perceptron_foundations import (
    Perceptron,
    plot_activation_functions,
    test_logic_gates
)
from src.loss_and_optimizers import (
    Losses,
    SGDOptimizer,
    AdamOptimizer,
    run_optimizer_comparison
)
from src.neural_network_scratch import (
    DenseLayer,
    NeuralNetwork,
    plot_decision_boundary
)


def experiment_1_solve_xor():
    """
    Demonstrates that while a single perceptron fails on XOR,
    a 2-layer Neural Network with ReLU activation effortlessly solves it.
    """
    print("\n" + "=" * 75)
    print(" EXPERIMENT 1: SOLVING THE XOR PROBLEM WITH A 2-LAYER NUMPY NEURAL NET")
    print("=" * 75)

    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    nn = NeuralNetwork()
    nn.add(DenseLayer(n_in=2, n_out=8, activation="relu", init_type="he"))
    nn.add(DenseLayer(n_in=8, n_out=1, activation="sigmoid", init_type="xavier"))

    optimizer = AdamOptimizer(learning_rate=0.08)
    nn.fit(X, y, epochs=250, batch_size=4, optimizer=optimizer, loss_type="binary_cross_entropy", verbose=False)

    preds = nn.forward(X)
    pred_bin = (preds >= 0.5).astype(int).ravel()

    print("Inputs X:")
    print(X)
    print(f"True XOR Targets:      {y.ravel().tolist()}")
    print(f"Neural Net Probabilities: {[round(float(p), 4) for p in preds.ravel()]}")
    print(f"Discrete Predictions:  {pred_bin.tolist()}")
    acc = np.mean(pred_bin == y.ravel())
    print(f"Final Accuracy:        {acc * 100:.1f}% -> {'SUCCESSFULLY SOLVED!' if acc == 1.0 else 'Incomplete'}")


def experiment_2_synthetic_decision_boundary():
    """
    Demonstrates non-linear classification boundary on the 2D 'Two Moons' dataset.
    """
    print("\n" + "=" * 75)
    print(" EXPERIMENT 2: NON-LINEAR DECISION BOUNDARY ON TWO MOONS DATASET")
    print("=" * 75)

    X, y = make_moons(n_samples=500, noise=0.20, random_state=42)
    y_col = y.reshape(-1, 1)

    nn = NeuralNetwork()
    nn.add(DenseLayer(n_in=2, n_out=16, activation="relu", init_type="he"))
    nn.add(DenseLayer(n_in=16, n_out=8, activation="relu", init_type="he"))
    nn.add(DenseLayer(n_in=8, n_out=1, activation="sigmoid", init_type="xavier"))

    optimizer = AdamOptimizer(learning_rate=0.02)
    nn.fit(X, y_col, epochs=80, batch_size=32, optimizer=optimizer, loss_type="binary_cross_entropy", verbose=False)

    acc = nn.compute_accuracy(y_col, nn.forward(X), loss_type="binary_cross_entropy")
    print(f"Two Moons Final Training Accuracy: {acc * 100:.2f}%")

    output_path = "outputs/moons_decision_boundary.png"
    plot_decision_boundary(nn, X, y, f"NumPy Neural Network Decision Boundary (Accuracy: {acc*100:.1f}%)", output_path)


def load_mnist_data(csv_path: str = "data/mnist_train.csv", max_train: int = 10000, max_val: int = 2000):
    """
    Loads MNIST dataset, scales pixel values from [0, 255] to [0.0, 1.0],
    and creates one-hot encoded label vectors.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"MNIST CSV not found at: {csv_path}")

    print(f"Loading MNIST from {csv_path}...")
    df = pd.read_csv(csv_path, nrows=(max_train + max_val))
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.astype(np.float32) / 255.0  # Normalize to [0, 1]

    # One-hot encode targets
    num_classes = 10
    y_one_hot = np.eye(num_classes)[y]

    X_train, X_val = X[:max_train], X[max_train:max_train + max_val]
    y_train, y_val = y_one_hot[:max_train], y_one_hot[max_train:max_train + max_val]
    y_val_raw = y[max_train:max_train + max_val]

    return X_train, y_train, X_val, y_val, y_val_raw


def experiment_3_mnist_benchmark():
    """
    Trains a Deep Neural Network from scratch on MNIST Digits.
    Compares Vanilla SGD vs. Adam Optimizer and generates comparative learning curves.
    """
    print("\n" + "=" * 75)
    print(" EXPERIMENT 3: REAL-WORLD MNIST DIGIT CLASSIFIER (SGD vs. ADAM)")
    print("=" * 75)

    X_train, y_train, X_val, y_val, y_val_raw = load_mnist_data(max_train=10000, max_val=2000)
    print(f"Train samples: {X_train.shape[0]} (784 features) | Validation samples: {X_val.shape[0]}")

    epochs = 15
    batch_size = 64

    # 1. Model with Vanilla SGD
    print("\n[Model 1] Training with Vanilla SGD (lr=0.05)...")
    nn_sgd = NeuralNetwork()
    nn_sgd.add(DenseLayer(n_in=784, n_out=128, activation="relu", init_type="he"))
    nn_sgd.add(DenseLayer(n_in=128, n_out=64, activation="relu", init_type="he"))
    nn_sgd.add(DenseLayer(n_in=64, n_out=10, activation="softmax", init_type="xavier"))

    t0 = time.time()
    hist_sgd = nn_sgd.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size,
        optimizer=SGDOptimizer(learning_rate=0.05),
        loss_type="categorical_cross_entropy",
        validation_data=(X_val, y_val),
        verbose=True
    )
    time_sgd = time.time() - t0
    final_val_acc_sgd = hist_sgd["val_acc"][-1]

    # 2. Model with Adam Optimizer
    print("\n[Model 2] Training with Adam Optimizer (lr=0.002)...")
    nn_adam = NeuralNetwork()
    nn_adam.add(DenseLayer(n_in=784, n_out=128, activation="relu", init_type="he"))
    nn_adam.add(DenseLayer(n_in=128, n_out=64, activation="relu", init_type="he"))
    nn_adam.add(DenseLayer(n_in=64, n_out=10, activation="softmax", init_type="xavier"))

    t0 = time.time()
    hist_adam = nn_adam.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size,
        optimizer=AdamOptimizer(learning_rate=0.002),
        loss_type="categorical_cross_entropy",
        validation_data=(X_val, y_val),
        verbose=True
    )
    time_adam = time.time() - t0
    final_val_acc_adam = hist_adam["val_acc"][-1]

    print("\n" + "-" * 75)
    print(f"BENCHMARK SUMMARY (over {epochs} epochs):")
    print(f"Vanilla SGD -> Validation Accuracy: {final_val_acc_sgd * 100:.2f}% | Training Time: {time_sgd:.2f}s")
    print(f"Adam        -> Validation Accuracy: {final_val_acc_adam * 100:.2f}% | Training Time: {time_adam:.2f}s")
    print("-" * 75)

    # Plot Training & Validation Curves Comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    epoch_axis = range(1, epochs + 1)

    # Loss Curves
    axes[0].plot(epoch_axis, hist_sgd["train_loss"], label="SGD Train Loss", color="red", linestyle="--")
    axes[0].plot(epoch_axis, hist_sgd["val_loss"], label="SGD Val Loss", color="darkred")
    axes[0].plot(epoch_axis, hist_adam["train_loss"], label="Adam Train Loss", color="blue", linestyle="--")
    axes[0].plot(epoch_axis, hist_adam["val_loss"], label="Adam Val Loss", color="navy")
    axes[0].set_title("Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Accuracy Curves
    axes[1].plot(epoch_axis, hist_sgd["train_acc"], label="SGD Train Acc", color="red", linestyle="--")
    axes[1].plot(epoch_axis, hist_sgd["val_acc"], label="SGD Val Acc", color="darkred")
    axes[1].plot(epoch_axis, hist_adam["train_acc"], label="Adam Train Acc", color="blue", linestyle="--")
    axes[1].plot(epoch_axis, hist_adam["val_acc"], label="Adam Val Acc", color="navy")
    axes[1].set_title("Classification Accuracy vs. Epochs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle("Scratch Neural Network on MNIST: SGD vs. Adam Optimizer", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plot_path = "outputs/mnist_sgd_vs_adam.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved MNIST optimizer comparison plot to: {plot_path}")

    # Plot Sample Visual Inferences
    plot_sample_predictions(nn_adam, X_val, y_val_raw, "outputs/mnist_sample_predictions.png")


def plot_sample_predictions(model: NeuralNetwork, X_val: np.ndarray, y_val_raw: np.ndarray, output_path: str):
    """
    Visualizes 5 test digits with their ground truth label, predicted class,
    and predicted confidence percentage.
    """
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_val), 5, replace=False)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    fig.suptitle("NumPy Neural Network: In-the-Wild Predictions on Test Digits", fontsize=14, fontweight="bold")

    for i, idx in enumerate(sample_indices):
        img = X_val[idx].reshape(28, 28)
        probs = model.forward(X_val[idx:idx + 1])[0]
        pred_label = int(np.argmax(probs))
        confidence = probs[pred_label] * 100.0
        true_label = int(y_val_raw[idx])

        color = "green" if pred_label == true_label else "red"
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)", color=color, fontweight="bold")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved sample digit predictions plot to: {output_path}")


def run_all():
    print("=" * 75)
    print(" WEEK 15 & 16: DEEP LEARNING & NUMPY NEURAL NETWORK SUITE")
    print("=" * 75)

    # 1. Logic gates and activations
    test_logic_gates()
    plot_activation_functions()

    # 2. Ravine optimizer comparison
    run_optimizer_comparison()

    # 3. Non-linear XOR solution
    experiment_1_solve_xor()

    # 4. Synthetic Moons boundary
    experiment_2_synthetic_decision_boundary()

    # 5. MNIST Real-World Digits Benchmark
    experiment_3_mnist_benchmark()

    print("\n" + "=" * 75)
    print(" ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(" All artifact plots saved to: outputs/")
    print("=" * 75)


if __name__ == "__main__":
    run_all()
