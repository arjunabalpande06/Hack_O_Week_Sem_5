import numpy as np
import pandas as pd
from dataset import GENRES, get_movie_matrix


def compute_covariance_matrix(X):
    """
    Computes covariance matrix C for centered feature matrix X_c.
    Formula: C = (1 / (N - 1)) * (X_c.T @ X_c)
    """
    mean_vec = np.mean(X, axis=0)
    X_c = X - mean_vec
    N = X.shape[0]
    C = (X_c.T @ X_c) / (N - 1)
    return mean_vec, X_c, C


def compute_eigen_decomposition(C):
    """
    Computes eigenvalues and eigenvectors for symmetric covariance matrix C.
    Solves equation: C * e = lambda * e
    
    Returns:
        eigenvalues_sorted (np.ndarray): Descending ordered eigenvalues
        eigenvectors_sorted (np.ndarray): Corresponding eigenvector columns
        explained_variance_ratio (np.ndarray): Variance proportion per eigenvector
    """
    # eigh is optimal for real symmetric matrices
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:, idx]
    
    # Variance ratio
    total_var = np.sum(eigenvalues_sorted)
    explained_variance_ratio = eigenvalues_sorted / total_var if total_var > 0 else np.zeros_like(eigenvalues_sorted)
    
    return eigenvalues_sorted, eigenvectors_sorted, explained_variance_ratio


def run_pca_scratch(n_components=2):
    """
    Applies PCA from scratch using Linear Algebra (Eigenvalues & Eigenvectors).
    
    Returns:
        titles (list): Movie titles
        Z (np.ndarray): Reduced feature matrix of shape (N, n_components)
        eigenvalues (np.ndarray): Eigenvalues of Covariance Matrix
        eigenvectors (np.ndarray): Eigenvectors of Covariance Matrix
        explained_variance_ratio (np.ndarray): Explained variance per PC
        C (np.ndarray): Covariance Matrix (d x d)
        X_c (np.ndarray): Centered Feature Matrix
    """
    titles, X = get_movie_matrix()
    mean_vec, X_c, C = compute_covariance_matrix(X)
    eigenvalues, eigenvectors, exp_var = compute_eigen_decomposition(C)
    
    # Select top k eigenvectors for projection matrix W
    W_k = eigenvectors[:, :n_components]  # Shape (d, k)
    
    # Projection: Z = X_c @ W_k
    Z = X_c @ W_k  # Shape (N, k)
    
    return {
        "titles": titles,
        "Z": Z,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "explained_variance_ratio": exp_var,
        "C": C,
        "X_c": X_c,
        "W_k": W_k,
        "mean_vec": mean_vec
    }
