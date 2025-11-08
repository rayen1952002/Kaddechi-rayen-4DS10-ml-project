import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from data_processor import DataProcessor, calculate_accuracy

class TestDataProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = DataProcessor()
    
    def test_generate_sample_data(self):
        """Test data generation"""
        X, y = self.processor.generate_sample_data(50)
        self.assertEqual(X.shape, (50, 3))
        self.assertEqual(len(y), 50)
    
    def test_split_data(self):
        """Test data splitting"""
        self.processor.generate_sample_data(100)
        X_train, X_test, y_train, y_test = self.processor.split_data(test_size=0.2)
        
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
    
    def test_calculate_accuracy(self):
        """Test accuracy calculation"""
        y_true = [1, 0, 1, 1, 0]
        y_pred = [1, 0, 1, 0, 0]
        
        accuracy = calculate_accuracy(y_true, y_pred)
        expected_accuracy = 0.8  # 4 out of 5 correct
        
        self.assertAlmostEqual(accuracy, expected_accuracy, places=6)

if __name__ == '__main__':
    unittest.main()
