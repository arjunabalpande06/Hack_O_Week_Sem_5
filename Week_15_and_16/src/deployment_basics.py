"""
Model Deployment Basics: Saving & Loading Models with Pickle and Joblib
Dataset: Heart Disease Classification (data/heart.csv)

This module demonstrates:
1. Data loading, preprocessing, and training a scikit-learn pipeline.
2. Model serialization using Python's built-in `pickle`.
3. Model serialization using `joblib` (optimized for NumPy arrays & ML models).
4. Performance benchmarks: serialization speed, file size, and loading latency.
5. Verification of prediction fidelity between original and loaded models.
6. A clean inference function for deployment simulation.
"""

import os
import time
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def load_heart_data(data_path: str = "data/heart.csv"):
    """Loads and splits the Heart Disease dataset into train and test sets."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def train_model_pipeline(X_train, y_train):
    """
    Creates and trains an end-to-end Machine Learning Pipeline.
    Bundling the scaler and classifier inside a Pipeline ensures identical
    preprocessing during deployment without feature leakage.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def benchmark_serialization(model, X_test, models_dir: str = "models"):
    """
    Saves and reloads the model using both Pickle and Joblib,
    measuring file size, save time, load time, and inference fidelity.
    """
    os.makedirs(models_dir, exist_ok=True)
    pickle_path = os.path.join(models_dir, "heart_pipeline_pickle.pkl")
    joblib_path = os.path.join(models_dir, "heart_pipeline_joblib.joblib")
    joblib_compressed_path = os.path.join(models_dir, "heart_pipeline_compressed.joblib")

    results = {}

    # 1. Pickle Serialization
    t0 = time.perf_counter()
    with open(pickle_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_save_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    with open(pickle_path, "rb") as f:
        pickle_model = pickle.load(f)
    pickle_load_time = time.perf_counter() - t0
    pickle_size_bytes = os.path.getsize(pickle_path)

    # 2. Joblib Serialization (Default)
    t0 = time.perf_counter()
    joblib.dump(model, joblib_path)
    joblib_save_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    joblib_model = joblib.load(joblib_path)
    joblib_load_time = time.perf_counter() - t0
    joblib_size_bytes = os.path.getsize(joblib_path)

    # 3. Joblib Serialization with Compression (zlib level 3)
    t0 = time.perf_counter()
    joblib.dump(model, joblib_compressed_path, compress=3)
    joblib_comp_save_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    joblib_comp_model = joblib.load(joblib_compressed_path)
    joblib_comp_load_time = time.perf_counter() - t0
    joblib_comp_size_bytes = os.path.getsize(joblib_compressed_path)

    # Verify Predictions match exactly
    original_preds = model.predict_proba(X_test)
    pickle_preds = pickle_model.predict_proba(X_test)
    joblib_preds = joblib_model.predict_proba(X_test)

    assert np.allclose(original_preds, pickle_preds), "Pickle predictions mismatch!"
    assert np.allclose(original_preds, joblib_preds), "Joblib predictions mismatch!"

    results["pickle"] = {
        "file": pickle_path,
        "size_kb": pickle_size_bytes / 1024,
        "save_time_ms": pickle_save_time * 1000,
        "load_time_ms": pickle_load_time * 1000,
        "verified": True
    }
    results["joblib"] = {
        "file": joblib_path,
        "size_kb": joblib_size_bytes / 1024,
        "save_time_ms": joblib_save_time * 1000,
        "load_time_ms": joblib_load_time * 1000,
        "verified": True
    }
    results["joblib_compressed"] = {
        "file": joblib_compressed_path,
        "size_kb": joblib_comp_size_bytes / 1024,
        "save_time_ms": joblib_comp_save_time * 1000,
        "load_time_ms": joblib_comp_load_time * 1000,
        "verified": True
    }

    return results, joblib_model


class HeartDiseaseDeploymentService:
    """
    Simulated Production Deployment Service.
    Loads the serialized model artifact once and handles incoming single-patient requests.
    """
    def __init__(self, model_path: str = "models/heart_pipeline_joblib.joblib", feature_names=None):
        self.model_path = model_path
        self.model = joblib.load(model_path)
        self.feature_names = feature_names or [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]

    def predict_single(self, patient_data: dict) -> dict:
        """
        Takes raw dictionary patient data, executes inference, and formats prediction output.
        """
        df_input = pd.DataFrame([patient_data])[self.feature_names]
        prob = self.model.predict_proba(df_input)[0, 1]
        prediction = int(prob >= 0.5)

        return {
            "prediction": prediction,
            "status": "High Risk: Heart Disease Detected" if prediction == 1 else "Normal / Low Risk",
            "heart_disease_probability": round(float(prob), 4),
            "confidence_score": round(float(prob if prediction == 1 else 1 - prob), 4)
        }


def run_deployment_demo():
    print("=" * 70)
    print(" WEEK 15 & 16: PART 1 - MODEL DEPLOYMENT BASICS (PICKLE & JOBLIB)")
    print("=" * 70)

    # 1. Load Data
    X_train, X_test, y_train, y_test, features = load_heart_data()
    print(f"Loaded dataset successfully. Features: {len(features)}")
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 2. Train Pipeline
    print("\nTraining StandardScaler + RandomForestClassifier Pipeline...")
    pipeline = train_model_pipeline(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    print(f"Model Performance -> Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f}")

    # 3. Benchmark Serialization
    print("\nBenchmarking Serialization (Pickle vs Joblib vs Joblib Compressed)...")
    benchmarks, saved_model = benchmark_serialization(pipeline, X_test)
    
    print("-" * 70)
    print(f"{'Method':<20} | {'File Size (KB)':<15} | {'Save Time (ms)':<15} | {'Load Time (ms)':<15}")
    print("-" * 70)
    for method, stats in benchmarks.items():
        print(f"{method:<20} | {stats['size_kb']:<15.2f} | {stats['save_time_ms']:<15.2f} | {stats['load_time_ms']:<15.2f}")
    print("-" * 70)

    # 4. Deployment Service Simulation
    print("\nInitializing Production Deployment Service from Joblib Artifact...")
    service = HeartDiseaseDeploymentService("models/heart_pipeline_joblib.joblib", features)

    # Sample Patient A (Diagnosis: Negative / No Disease, Target = 0)
    sample_patient_neg = {
        'age': 52, 'sex': 1, 'cp': 0, 'trestbps': 125, 'chol': 212,
        'fbs': 0, 'restecg': 1, 'thalach': 168, 'exang': 0,
        'oldpeak': 1.0, 'slope': 2, 'ca': 2, 'thal': 3
    }
    # Sample Patient B (Diagnosis: Positive / Heart Disease, Target = 1)
    sample_patient_pos = {
        'age': 58, 'sex': 0, 'cp': 0, 'trestbps': 100, 'chol': 248,
        'fbs': 0, 'restecg': 0, 'thalach': 122, 'exang': 0,
        'oldpeak': 1.0, 'slope': 1, 'ca': 0, 'thal': 2
    }

    print("\n[Inference 1 - Patient Diagnosis Test (Expected: Target 0)]:")
    res1 = service.predict_single(sample_patient_neg)
    print(res1)

    print("\n[Inference 2 - Patient Diagnosis Test (Expected: Target 1)]:")
    res2 = service.predict_single(sample_patient_pos)
    print(res2)
    print("=" * 70)


if __name__ == "__main__":
    run_deployment_demo()
