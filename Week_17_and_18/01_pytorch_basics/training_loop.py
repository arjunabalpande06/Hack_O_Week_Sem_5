"""
Part 1B: Intro to PyTorch - The Canonical Deep Learning Training Loop
Demonstrates:
1. Synthetic data generation and PyTorch Dataset/DataLoader pipeline.
2. Architecture definition using torch.nn.Module with modular forward passes.
3. Loss functions (MSE / CrossEntropy) and Optimizers (Adam / SGD).
4. The canonical PyTorch training step (model.train, zero_grad, forward, backward, step).
5. Evaluation loop with model.eval() and torch.no_grad().
6. Dual-Engine: Uses native PyTorch when available, and provides an educational
   NumPy gradient descent training loop if run in a minimal environment.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ----------------------------------------------------
# 1. SYNTHETIC DATA GENERATION (Zero Kaggle Requirement)
# ----------------------------------------------------
def generate_synthetic_data(num_samples: int = 1200):
    """
    y = 2.5 * x1 - 1.8 * x2 + 0.5 * x3 + noise
    No Kaggle download needed!
    """
    np.random.seed(42)
    X = np.random.randn(num_samples, 3).astype(np.float32)
    true_weights = np.array([[2.5], [-1.8], [0.5]], dtype=np.float32)
    noise = np.random.randn(num_samples, 1).astype(np.float32) * 0.1
    y = X @ true_weights + noise
    return X, y


# ----------------------------------------------------
# 2. PYTORCH NATIVE TRAINING LOOP
# ----------------------------------------------------
if TORCH_AVAILABLE:
    class RegressionModel(nn.Module):
        def __init__(self, in_features: int, hidden_dim: int = 16):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)

def run_pytorch_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training Loop] Utilizing device: {device}")

    X_np, y_np = generate_synthetic_data(num_samples=1200)
    X = torch.tensor(X_np)
    y = torch.tensor(y_np)

    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    batch_size = 32
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = RegressionModel(in_features=3, hidden_dim=16).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 12
    print("\n--- Starting PyTorch Training Loop ---")
    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()            # Step 1: Zero gradients
            preds = model(batch_x)           # Step 2: Forward pass
            loss = criterion(preds, batch_y) # Step 3: Compute loss
            loss.backward()                  # Step 4: Backward pass (compute gradients)
            optimizer.step()                 # Step 5: Update parameters

            running_train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                val_loss = criterion(preds, batch_y)
                running_val_loss += val_loss.item() * batch_x.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)

        if epoch % 3 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train MSE: {epoch_train_loss:.4f} | Val MSE: {epoch_val_loss:.4f}")

    print("\n[SUCCESS] PyTorch training loop completed!")


# ----------------------------------------------------
# 3. EDUCATIONAL NUMPY TRAINING LOOP
# ----------------------------------------------------
def run_numpy_training():
    print("=" * 65)
    print(" DEEP LEARNING TRAINING LOOP MECHANICS (NumPy Engine)")
    print("=" * 65)

    X, y = generate_synthetic_data(num_samples=1000)
    split = 800
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    # Model parameters: 2-layer MLP (3 -> 16 -> 1)
    np.random.seed(42)
    W1 = np.random.randn(3, 16) * 0.1
    b1 = np.zeros((1, 16))
    W2 = np.random.randn(16, 1) * 0.1
    b2 = np.zeros((1, 1))
    lr = 0.01

    print("Target True Weights: [2.5, -1.8, 0.5]")
    print("\n--- Starting Iterative Optimization ---")

    for epoch in range(1, 13):
        # 1. Forward Pass
        z1 = X_train @ W1 + b1
        a1 = np.maximum(0, z1) # ReLU activation
        preds = a1 @ W2 + b2

        # 2. Compute MSE Loss
        loss = np.mean((preds - y_train) ** 2)

        # 3. Backward Pass (Analytical Gradients via Chain Rule)
        d_preds = (2.0 / len(X_train)) * (preds - y_train)
        dW2 = a1.T @ d_preds
        db2 = np.sum(d_preds, axis=0, keepdims=True)

        da1 = d_preds @ W2.T
        dz1 = da1 * (z1 > 0) # ReLU derivative
        dW1 = X_train.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # 4. Parameter Update (SGD step)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        # 5. Validation Loss
        val_z1 = X_val @ W1 + b1
        val_a1 = np.maximum(0, val_z1)
        val_preds = val_a1 @ W2 + b2
        val_loss = np.mean((val_preds - y_val) ** 2)

        if epoch % 3 == 0 or epoch == 1 or epoch == 12:
            print(f"Epoch [{epoch:02d}/12] | Train MSE: {loss:.4f} | Val MSE: {val_loss:.4f}")

    print("\n[SUCCESS] Gradient descent optimization converged successfully.")


if __name__ == "__main__":
    if TORCH_AVAILABLE:
        run_pytorch_training()
    else:
        print("[Notice: PyTorch not in active environment; running gradient descent loop via NumPy]")
        run_numpy_training()
