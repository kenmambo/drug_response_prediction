import os
import pandas as pd
import numpy as np

def create_mock_datasets(data_dir="data"):
    """Create mock datasets for drug response prediction.
    
    Args:
        data_dir (str): Directory where mock data will be saved
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Create mock cell lines data
        print("Creating mock cell line data...")
        celllines_data = {
            'Cell_Line_ID': list(range(1, 101)),
            'Cell_Line_Name': [f'CellLine_{i}' for i in range(1, 101)],
            'Tissue': np.random.choice(['Lung', 'Breast', 'Colon', 'Skin', 'Blood'], 100),
            'Cancer_Type': np.random.choice(['NSCLC', 'SCLC', 'Melanoma', 'Carcinoma'], 100)
        }
        celllines_df = pd.DataFrame(celllines_data)
        celllines_df.to_csv(os.path.join(data_dir, "gdsc_celllines.csv"), index=False)
        
        # Create mock gene expression data
        print("Creating mock gene expression data...")
        genes = [f'Gene_{i}' for i in range(1, 1001)]
        expression_data = {'Cell_Line_ID': list(range(1, 101))}
        for gene in genes:
            expression_data[gene] = np.random.normal(0, 1, 100)
        expression_df = pd.DataFrame(expression_data)
        expression_df.to_csv(os.path.join(data_dir, "gdsc_expression.csv"), index=False)
        
        # Create mock drug response data
        print("Creating mock drug response data...")
        drug_response_data = {
            'Cell_Line_ID': list(range(1, 101)),
            'Drug_ID': np.random.randint(1, 10, 100),
            'Drug_Name': ['5-Fluorouracil'] * 100,
            'IC50': np.random.lognormal(0, 1, 100)
        }
        drug_response_df = pd.DataFrame(drug_response_data)
        drug_response_df.to_csv(os.path.join(data_dir, "gdsc_drug_response.csv"), index=False)
        
        print("Mock datasets created successfully!")
        return True
    except Exception as e:
        print(f"Error creating mock datasets: {e}")
        return False

# Run the function if this script is executed directly
if __name__ == "__main__":
    create_mock_datasets()