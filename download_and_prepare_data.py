import os
import requests
import pandas as pd
import numpy as np
import time

# Create data directory
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(data_dir, exist_ok=True)
print(f"Data will be saved to: {data_dir}")


def download_data():
    """Attempt to download data from cancerrxgene.org"""
    success = True
    
    # URLs for data download
    urls = {
        "gdsc_celllines.csv": "https://www.cancerrxgene.org/api/celllines?download=true",
        "gdsc_expression.csv": "https://www.cancerrxgene.org/api/expression?download=true",
        "gdsc_drug_response.csv": "https://www.cancerrxgene.org/api/drug_response?download=true"
    }
    
    for filename, url in urls.items():
        try:
            print(f"Downloading {filename}...")
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            # Verify file was created and has content
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                print(f"✓ Successfully downloaded {filename}")
            else:
                print(f"✗ File {filename} was created but appears to be empty")
                success = False
        except Exception as e:
            print(f"✗ Error downloading {filename}: {e}")
            success = False
    
    return success


def create_mock_data():
    """Create mock datasets for drug response prediction"""
    try:
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
        
        print("✓ Mock datasets created successfully!")
        return True
    except Exception as e:
        print(f"✗ Error creating mock datasets: {e}")
        return False


def verify_data_files():
    """Verify that all required data files exist and have content"""
    required_files = ["gdsc_celllines.csv", "gdsc_expression.csv", "gdsc_drug_response.csv"]
    all_exist = True
    
    for filename in required_files:
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"✗ Missing or empty file: {filename}")
            all_exist = False
        else:
            print(f"✓ Found file: {filename}")
    
    return all_exist


# Main execution
print("=== Drug Response Prediction Data Preparation ===")

# First try to download the data
print("\nAttempting to download data from cancerrxgene.org...")
download_success = download_data()

# Verify if download was successful
if download_success:
    print("\nVerifying downloaded data...")
    if verify_data_files():
        print("\n✓ All data files downloaded successfully!")
    else:
        print("\n✗ Some downloaded files are missing or empty. Creating mock data instead...")
        create_mock_data()
else:
    print("\n✗ Download failed. Creating mock data instead...")
    create_mock_data()

# Final verification
print("\nFinal data verification:")
if verify_data_files():
    print("\n✓ Data preparation complete! You can now proceed with data preprocessing and model training.")
else:
    print("\n✗ Data preparation failed. Please check the errors above and try again.")