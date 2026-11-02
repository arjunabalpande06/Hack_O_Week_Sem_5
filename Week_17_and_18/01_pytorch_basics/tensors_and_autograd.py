"""
Part 1A: Intro to PyTorch - Tensors & Autograd
Demonstrates:
1. Tensor creation, data types, and device management (CPU & CUDA GPU).
2. Fundamental tensor operations (reshaping, broadcasting, matrix multiplication).
3. The Autograd engine: computational graph, requires_grad, backward pass, and gradient accumulation.
4. Analytical verification of gradients against calculus.
5. Dual-Engine: Uses native PyTorch when available, and provides an educational
   computational graph engine if run in a minimal environment.
"""

import sys
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def section_header(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


# ======================================================================
# PYTORCH NATIVE IMPLEMENTATION
# ======================================================================
def run_pytorch_native():
    section_header("1. TENSOR CREATION & PROPERTIES (PyTorch)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    t_list = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    print(f"Tensor from list:\n{t_list} | Shape: {t_list.shape} | Dtype: {t_list.dtype}")

    zeros = torch.zeros((2, 3), dtype=torch.float32)
    ones = torch.ones((2, 3), dtype=torch.float32)
    rand_normal = torch.randn((2, 3))
    lin = torch.linspace(0, 10, steps=5)

    print(f"Zeros (2, 3):\n{zeros}")
    print(f"Standard Normal (2, 3):\n{rand_normal}")
    print(f"Linspace [0, 10, 5]: {lin}")

    section_header("2. TENSOR OPERATIONS & MANIPULATION")
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    print("Matrix A @ B:\n", A @ B)

    x = torch.arange(12).view(3, 4)
    print(f"Reshaped (12,) -> (3, 4):\n{x}")

    section_header("3. AUTOGRAD & COMPUTATIONAL GRAPH")
    # Let y = 3*x^2 + 2*x + 1
    # dy/dx = 6*x + 2 -> at x=4: 6(4) + 2 = 26
    x_val = torch.tensor(4.0, requires_grad=True)
    y = 3 * (x_val ** 2) + 2 * x_val + 1
    print(f"Input x: {x_val.item()} | Computed y: {y.item()} | grad_fn: {y.grad_fn}")

    y.backward()
    print(f"Autograd dy/dx:   {x_val.grad.item()}")
    print(f"Analytical dy/dx: {6.0 * 4.0 + 2.0}")

    # Gradient accumulation demonstration
    y2 = 3 * (x_val ** 2) + 2 * x_val + 1
    y2.backward()
    print(f"Accumulated x.grad without zeroing: {x_val.grad.item()} (26 + 26 = 52)")
    x_val.grad.zero_()
    print(f"After x_val.grad.zero_(): {x_val.grad.item()}")


# ======================================================================
# EDUCATIONAL COMPUTATIONAL GRAPH & AUTOGRAD ENGINE (NumPy / Pure Python)
# ======================================================================
class Value:
    """Scalar Autograd node tracking the computational graph for reverse-mode automatic differentiation."""
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, power):
        out = Value(self.data ** power, (self,), f"**{power}")
        def _backward():
            self.grad += (power * (self.data ** (power - 1))) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # Topological sort of computational graph DAG
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def run_educational_autograd():
    section_header("1. TENSORS & ARRAYS MECHANICS (NumPy Engine)")
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    print("Matrix A (2x2):\n", A)
    print("Matrix B (2x2):\n", B)
    print("Matrix Multiplication A @ B:\n", A @ B)

    section_header("2. AUTOGRAD & COMPUTATIONAL GRAPH (Reverse-Mode AD)")
    # Target function: y = 3 * x^2 + 2 * x + 1 at x = 4.0
    # dy/dx = 6*x + 2 = 26.0
    x = Value(4.0)
    c3 = Value(3.0)
    c2 = Value(2.0)
    c1 = Value(1.0)

    # Forward pass builds directed acyclic graph (DAG)
    y = (c3 * (x ** 2)) + (c2 * x) + c1
    print(f"Input x: {x.data}")
    print(f"Forward pass output y: {y.data}")

    # Backward pass traverses DAG via chain rule
    y.backward()
    print(f"Computed Autograd dy/dx:   {x.grad:.4f}")
    print(f"Analytical Calculus dy/dx: {6.0 * 4.0 + 2.0:.4f}")
    assert abs(x.grad - 26.0) < 1e-5
    print("-> Autograd derivative verified to exactly match calculus!")


if __name__ == "__main__":
    if TORCH_AVAILABLE:
        print("[Running PyTorch Native Implementation]")
        run_pytorch_native()
    else:
        print("[Notice: PyTorch not in active environment; running full computational graph engine]")
        run_educational_autograd()
