import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from data_processor import DataProcessor, calculate_accuracy

class TestDataProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = DataProcessor()

    def test_generate_sample_data(self):
        """Test data generation with new processor"""
        X, y = self.processor.generate_sample_data(50, 3)  # Specify 3 features to match old test
        self.assertEqual(X.shape, (50, 3))
        self.assertEqual(len(y), 50)

    def test_calculate_accuracy(self):
        """Test accuracy calculation"""
        y_true = [1, 0, 1, 1, 0]
        y_pred = [1, 0, 1, 0, 0]
        accuracy = calculate_accuracy(y_true, y_pred)
        expected_accuracy = 0.8
        self.assertAlmostEqual(accuracy, expected_accuracy, places=6)

    def test_model_training(self):
        """Test model training functionality"""
        X, y = self.processor.generate_sample_data(100, 3)
        model = self.processor.train_model(n_estimators=10, max_depth=3)
        self.assertIsNotNone(model)
        
    def test_prediction(self):
        """Test prediction functionality"""
        X, y = self.processor.generate_sample_data(50, 3)
        self.processor.train_model(n_estimators=10, max_depth=3)
        predictions = self.processor.predict(X[:5])
        self.assertEqual(len(predictions), 5)
