import os
import requests
import pandas as pd
import sys
import time

# Determine the correct data directory path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")

# Create data directory
os.makedirs(data_dir, exist_ok=True)
print(f"Data will be saved to: {data_dir}")

# Function to download data with error handling
def download_file(url, filename, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"Downloading {filename}... (Attempt {attempt+1}/{max_retries})")
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            # Verify file was created and has content
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                print(f"✓ Successfully downloaded {filename}")
                return True
            else:
                print(f"✗ File {filename} was created but appears to be empty")
        except requests.exceptions.RequestException as e:
            print(f"✗ Error downloading {filename}: {e}")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
        
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    # If we get here, all attempts failed
    print(f"Failed to download {filename} after {max_retries} attempts")
    return False

# Download GDSC dataset (drug sensitivity)
gdsc_url = "https://www.cancerrxgene.org/api/celllines?download=true"
gdsc_success = download_file(gdsc_url, "gdsc_celllines.csv")

# Download gene expression data
expr_url = "https://www.cancerrxgene.org/api/expression?download=true"
expr_success = download_file(expr_url, "gdsc_expression.csv")

# Download drug response data
drug_url = "https://www.cancerrxgene.org/api/drug_response?download=true"
drug_success = download_file(drug_url, "gdsc_drug_response.csv")

# Check if all downloads were successful
if gdsc_success and expr_success and drug_success:
    print("\n✓ All datasets downloaded successfully!")
else:
    print("\n✗ Some downloads failed. Using create_mock_data.py as fallback...")
    # Import and run create_mock_data if it exists
    try:
        sys.path.append(project_root)
        from create_mock_data import create_mock_datasets
        create_mock_datasets()
        print("✓ Mock datasets created successfully as fallback!")
    except ImportError:
        print("✗ Could not import create_mock_data. Please run create_mock_data.py manually.")
    except Exception as e:
        print(f"✗ Error creating mock data: {e}")
        print("Please run create_mock_data.py manually.")