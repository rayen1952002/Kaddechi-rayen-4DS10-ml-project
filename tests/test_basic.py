import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

class TestBasicImports(unittest.TestCase):
    
    def test_import_numpy(self):
        """Test that numpy can be imported"""
        try:
            import numpy as np
            self.assertTrue(True)
        except ImportError:
            self.fail("Failed to import numpy")
    
    def test_import_pandas(self):
        """Test that pandas can be imported"""
        try:
            import pandas as pd
            self.assertTrue(True)
        except ImportError:
            self.fail("Failed to import pandas")
    
    def test_import_sklearn(self):
        """Test that sklearn can be imported"""
        try:
            import sklearn
            self.assertTrue(True)
        except ImportError:
            self.fail("Failed to import sklearn")

if __name__ == '__main__':
    unittest.main()
