import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
from typing import Tuple, Optional, Dict, Any
import warnings
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Enhanced Data Processor with MLflow tracking and model persistence
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.data = None
        self.model = None
        self.feature_names = ['feature_1', 'feature_2', 'feature_3']
        self.target_name = 'target'
        self._is_data_loaded = False
        self._is_model_trained = False
        np.random.seed(self.random_state)
        
        # MLflow setup
        mlflow.set_tracking_uri("file:./mlruns")  # Local storage
        logger.info(f"DataProcessor initialized with random_state={random_state}")

    def generate_sample_data(self, n_samples: int = 1000, n_features: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic sample data for ML training"""
        if n_samples <= 0 or n_features <= 0:
            raise ValueError("n_samples and n_features must be positive integers")
        
        logger.info(f"Generating {n_samples} samples with {n_features} features")
        
        # Generate more realistic data
        X = np.random.randn(n_samples, n_features)
        # Create meaningful relationships between features
        X[:, 0] = X[:, 0] * 1.5 + 2  # Shift and scale
        X[:, 1] = X[:, 1] * 0.8 + X[:, 0] * 0.3  # Correlation
        
        # Create meaningful target based on feature combinations
        coefficients = np.random.randn(n_features)
        linear_combination = np.dot(X, coefficients)
        probability = 1 / (1 + np.exp(-linear_combination))  # Sigmoid
        y = (probability > 0.5).astype(int)
        
        self.data = (X, y)
        self._is_data_loaded = True
        self.feature_names = [f'feature_{i+1}' for i in range(n_features)]
        
        logger.info(f"Data generated: X{X.shape}, y{y.shape}")
        return X, y

    def train_model(self, n_estimators: int = 100, max_depth: int = 5) -> RandomForestClassifier:
        """Train a RandomForest model with MLflow tracking"""
        if not self._is_data_loaded:
            raise ValueError("No data available. Call generate_sample_data first.")
        
        X, y = self.data
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("n_samples", X.shape[0])
            mlflow.log_param("n_features", X.shape[1])
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state
            )
            self.model.fit(X_train, y_train)
            
            # Predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Log model
            mlflow.sklearn.log_model(self.model, "random_forest_model")
            
            # Log feature importance
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            mlflow.log_dict(feature_importance, "feature_importance.json")
            
            logger.info(f"Model trained - Accuracy: {accuracy:.4f}")
            
        self._is_model_trained = True
        return self.model

    def save_model(self, filepath: str = "models/random_forest_model.pkl") -> None:
        """Save trained model as .pkl file"""
        if not self._is_model_trained:
            raise ValueError("No trained model available. Call train_model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Log model path in MLflow
        with mlflow.start_run():
            mlflow.log_artifact(filepath)

    def load_model(self, filepath: str = "models/random_forest_model.pkl") -> RandomForestClassifier:
        """Load model from .pkl file"""
        self.model = joblib.load(filepath)
        self._is_model_trained = True
        logger.info(f"Model loaded from {filepath}")
        return self.model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained model"""
        if not self._is_model_trained:
            raise ValueError("No trained model available.")
        return self.model.predict(X)

    def get_data_summary(self) -> dict:
        """Get comprehensive data summary"""
        if not self._is_data_loaded:
            return {
                "status": "no_data",
                "message": "No data loaded. Call generate_sample_data first."
            }
        
        try:
            X, y = self.data
            
            # Convert numpy arrays to Python lists for JSON serialization
            feature_stats = {}
            if X.size > 0:
                feature_stats = {
                    'means': [float(x) for x in X.mean(axis=0)],
                    'std_devs': [float(x) for x in X.std(axis=0)],
                    'min_values': [float(x) for x in X.min(axis=0)],
                    'max_values': [float(x) for x in X.max(axis=0)]
                }
            
            summary = {
                'n_samples': int(X.shape[0]),
                'n_features': int(X.shape[1]),
                'feature_names': self.feature_names,
                'target_distribution': {
                    'class_0': int(np.sum(y == 0)),
                    'class_1': int(np.sum(y == 1)),
                    'balance_ratio': float(np.sum(y == 1) / len(y)) if len(y) > 0 else 0
                },
                'feature_stats': feature_stats,
                'model_trained': self._is_model_trained
            }
            
            return summary
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error generating summary: {str(e)}"
            }

def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy score"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)

def calculate_precision(y_true, y_pred):
    """Calculate precision score"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0
