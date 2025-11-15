import unittest
import sys
import os
import numpy as np
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from data_processor import DataProcessor

class TestEnhancedDataProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = DataProcessor(random_state=42)
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_mlflow_integration(self):
        """Test MLflow model training and tracking"""
        X, y = self.processor.generate_sample_data(100, 3)
        model = self.processor.train_model(n_estimators=10, max_depth=3)
        
        self.assertIsNotNone(model)
        self.assertTrue(self.processor._is_model_trained)
    
    def test_model_persistence(self):
        """Test model saving and loading"""
        X, y = self.processor.generate_sample_data(100, 3)
        self.processor.train_model(n_estimators=10, max_depth=3)
        
        # Save model
        model_path = os.path.join(self.temp_dir, "test_model.pkl")
        self.processor.save_model(model_path)
        self.assertTrue(os.path.exists(model_path))
        
        # Load model
        new_processor = DataProcessor()
        new_processor.load_model(model_path)
        predictions = new_processor.predict(X[:5])
        
        self.assertEqual(len(predictions), 5)
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent after save/load"""
        X, y = self.processor.generate_sample_data(50, 3)
        self.processor.train_model(n_estimators=10, max_depth=3)
        
        original_predictions = self.processor.predict(X[:10])
        
        # Save and load
        model_path = os.path.join(self.temp_dir, "test_model.pkl")
        self.processor.save_model(model_path)
        
        new_processor = DataProcessor()
        new_processor.load_model(model_path)
        new_predictions = new_processor.predict(X[:10])
        
        np.testing.assert_array_equal(original_predictions, new_predictions)

class TestMetrics(unittest.TestCase):
    
    def test_accuracy_calculation(self):
        from data_processor import calculate_accuracy
        
        y_true = [1, 0, 1, 1, 0]
        y_pred = [1, 0, 1, 0, 0]
        
        accuracy = calculate_accuracy(y_true, y_pred)
        self.assertAlmostEqual(accuracy, 0.8, places=6)
    
    def test_precision_calculation(self):
        from data_processor import calculate_precision
        
        y_true = [1, 0, 1, 1, 0]
        y_pred = [1, 0, 1, 0, 0]
        
        precision = calculate_precision(y_true, y_pred)
        self.assertAlmostEqual(precision, 1.0, places=6)
