import numpy as np
import pandas as pd
from dataset import GENRES, get_movie_matrix, get_movie_vector


def dot_product(v1, v2):
    """
    Computes dot product between two vectors v1 and v2.
    Formula: v1 . v2 = sum(v1[i] * v2[i])
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    return float(np.dot(v1, v2))


def cosine_similarity(v1, v2):
    """
    Computes cosine similarity between two vectors v1 and v2.
    Formula: cos(theta) = (v1 . v2) / (||v1|| * ||v2||)
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def get_all_pairwise_similarities(use_cosine=True):
    """
    Computes full similarity matrix using matrix operations.
    If Cosine: S = X_norm @ X_norm.T
    If Dot Product: S = X @ X.T
    Returns:
        titles (list): Movie titles
        S (np.ndarray): N x N similarity matrix
    """
    titles, X = get_movie_matrix()
    if use_cosine:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        X_norm = X / norms
        S = np.matmul(X_norm, X_norm.T)
    else:
        S = np.matmul(X, X.T)
    return titles, S


def recommend_similar_movies(selected_title, top_n=5, metric="cosine"):
    """
    Recommends top N movies similar to selected_title based on specified metric.
    
    Args:
        selected_title (str): Title of selected movie
        top_n (int): Number of recommendations to return
        metric (str): 'cosine' or 'dot'
        
    Returns:
        recommendations (pd.DataFrame): DataFrame with recommended movies & similarity scores
        target_vector (np.ndarray): Feature vector of target movie
        breakdown_list (list): Step-by-step elementwise dot product math breakdown
    """
    titles, X = get_movie_matrix()
    target_idx = None
    for i, t in enumerate(titles):
        if t.lower() == selected_title.lower():
            target_idx = i
            break
            
    if target_idx is None:
        raise ValueError(f"Movie '{selected_title}' not found in matrix.")

    target_title = titles[target_idx]
    target_vector = X[target_idx]

    scores = []
    breakdowns = []

    for i, (title, vector) in enumerate(zip(titles, X)):
        if i == target_idx:
            continue  # Exclude self
            
        dp = dot_product(target_vector, vector)
        cs = cosine_similarity(target_vector, vector)
        score = cs if metric == "cosine" else dp
        
        # Calculate elementwise products for transparency/education
        elem_prod = target_vector * vector
        
        scores.append({
            "Title": title,
            "Similarity Score": round(score, 4),
            "Dot Product": round(dp, 4),
            "Cosine Sim": round(cs, 4),
            "Vector": vector,
            "ElemProduct": elem_prod,
            "Index": i
        })

    # Sort descending by score
    scores.sort(key=lambda x: x["Similarity Score"], reverse=True)
    top_recs = scores[:top_n]

    # Convert to DataFrame
    rec_df = pd.DataFrame([
        {
            "Rank": r + 1,
            "Movie": item["Title"],
            "Similarity Score": item["Similarity Score"],
            "Dot Product": item["Dot Product"],
            "Cosine Similarity": item["Cosine Sim"]
        }
        for r, item in enumerate(top_recs)
    ])

    return rec_df, target_title, target_vector, top_recs
