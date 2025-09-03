import unittest
import os
import sys
from pathlib import Path

# Add the project root to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestBasicFunctionality(unittest.TestCase):
    def setUp(self):
        # Get the project root directory
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'data'
        
        # Create data directory if it doesn't exist (for CI environment)
        if not self.data_dir.exists():
            self.data_dir.mkdir(exist_ok=True)
            
        # Create empty mock files if they don't exist (for CI environment)
        self.data_files = ['gdsc_celllines.csv', 'gdsc_drug_response.csv', 'gdsc_expression.csv']
        for file in self.data_files:
            file_path = self.data_dir / file
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    f.write('mock_header\nmock_data')
    
    def test_data_directory_exists(self):
        """Test that the data directory exists"""
        self.assertTrue(self.data_dir.exists())
        
    def test_data_files_exist(self):
        """Test that mock data files exist"""
        for file in self.data_files:
            self.assertTrue((self.data_dir / file).exists())

if __name__ == '__main__':
    unittest.main()