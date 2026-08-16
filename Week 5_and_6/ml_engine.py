import numpy as np
import pandas as pd
from dataset import GENRES, get_movie_matrix


class UserPreferenceLearner:
    """
    Learns user preference weight vector w and bias b using Gradient Descent.
    Demonstrates:
    - Derivative of Loss function: dL/d(y_hat) = y_hat - y
    - Gradient vector: grad_w L = (y_hat - y) * v_i
    - Chain Rule: dL/dw_j = (dL/dy_hat) * (dy_hat/dw_j)
    """

    def __init__(self, n_features=8, learning_rate=0.05):
        self.n_features = n_features
        self.lr = learning_rate
        # Initialize weights around neutral value (e.g., 0.5) or zero
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 3.0  # Start around neutral rating scale (1-5 scale center)
        self.history = []

    def predict(self, v):
        """
        Predicts rating for feature vector v.
        Formula: y_hat = w . v + b
        """
        return float(np.dot(self.w, v) + self.b)

    def train_step(self, rated_movies, epochs=100):
        """
        Runs Gradient Descent over rated_movies.
        
        Args:
            rated_movies (list of dict): [{"title": ..., "vector": ..., "rating": float}]
            epochs (int): Number of SGD / Batch GD iterations
            
        Returns:
            history (list): Log of losses, gradients, and weights per epoch
        """
        self.history = []
        N = len(rated_movies)
        if N == 0:
            return self.history

        vectors = np.array([m["vector"] for m in rated_movies], dtype=np.float64)
        ratings = np.array([m["rating"] for m in rated_movies], dtype=np.float64)

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            grad_w_total = np.zeros(self.n_features, dtype=np.float64)
            grad_b_total = 0.0

            epoch_details = []

            for i in range(N):
                v_i = vectors[i]
                y_i = ratings[i]

                # Step 1: Forward Pass (Dot Product)
                y_hat = np.dot(self.w, v_i) + self.b

                # Step 2: Error computation
                error = y_hat - y_i
                loss_i = 0.5 * (error ** 2)
                total_loss += loss_i

                # Step 3: Derivative & Chain Rule
                # dL/dy_hat = error
                # dy_hat/dw_j = v_ij
                # dL/dw_j = error * v_ij
                grad_w_i = error * v_i
                grad_b_i = error

                grad_w_total += grad_w_i
                grad_b_total += grad_b_i

                epoch_details.append({
                    "movie": rated_movies[i]["title"],
                    "y_true": y_i,
                    "y_hat": y_hat,
                    "error": error,
                    "grad_w": grad_w_i
                })

            # Average loss & gradients over batch
            avg_loss = total_loss / N
            avg_grad_w = grad_w_total / N
            avg_grad_b = grad_b_total / N

            # Parameter Update using Gradient Descent
            self.w -= self.lr * avg_grad_w
            self.b -= self.lr * avg_grad_b

            self.history.append({
                "epoch": epoch,
                "loss": avg_loss,
                "weights": self.w.copy(),
                "bias": self.b,
                "grad_w": avg_grad_w.copy(),
                "grad_b": avg_grad_b,
                "sample_details": epoch_details
            })

        return self.history

    def recommend_unrated(self, rated_titles, top_n=5):
        """
        Predicts ratings for all unrated movies in dataset and sorts top N recommendations.
        """
        titles, X = get_movie_matrix()
        rated_set = set([t.lower() for t in rated_titles])

        predictions = []
        for title, v in zip(titles, X):
            if title.lower() in rated_set:
                continue  # Skip already rated movies
                
            pred_rating = self.predict(v)
            # Clip predicted rating to realistic range [1, 5]
            clipped_rating = float(np.clip(pred_rating, 1.0, 5.0))
            
            predictions.append({
                "Movie": title,
                "Predicted Rating": round(clipped_rating, 2),
                "Raw Score": round(pred_rating, 3),
                "Vector": v
            })

        predictions.sort(key=lambda x: x["Predicted Rating"], reverse=True)
        rec_df = pd.DataFrame(predictions[:top_n])
        return rec_df
