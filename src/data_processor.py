import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataProcessor:
    """Simple data processor for testing"""

    def __init__(self):
        self.data = None

    def generate_sample_data(self, n_samples=100):
        """Generate sample data for testing"""
        np.random.seed(42)
        X = np.random.randn(n_samples, 3)
        y = np.random.randint(0, 2, n_samples)
        self.data = (X, y)
        return X, y

    def split_data(self, test_size=0.2):
        """Split data into train and test sets"""
        if self.data is None:
            raise ValueError("No data available. Call generate_sample_data first.")

        X, y = self.data
        return train_test_split(X, y, test_size=test_size, random_state=42)

def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy score"""
    # Convert to numpy arrays for proper element-wise comparison
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)
