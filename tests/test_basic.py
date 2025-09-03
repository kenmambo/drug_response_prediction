import unittest
import os

class TestBasicFunctionality(unittest.TestCase):
    def test_data_directory_exists(self):
        """Test that the data directory exists"""
        self.assertTrue(os.path.exists('data'))
        
    def test_data_files_exist(self):
        """Test that mock data files exist"""
        data_files = ['gdsc_celllines.csv', 'gdsc_drug_response.csv', 'gdsc_expression.csv']
        for file in data_files:
            self.assertTrue(os.path.exists(os.path.join('data', file)))

if __name__ == '__main__':
    unittest.main()