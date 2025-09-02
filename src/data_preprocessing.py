import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from Bio import SeqIO
from gseapy import enrichr
import gseapy as gp

def load_data():
    """Load all datasets"""
    # Load datasets
    celllines = pd.read_csv('data/gdsc_celllines.csv')
    expression = pd.read_csv('data/gdsc_expression.csv')
    drug_response = pd.read_csv('data/gdsc_drug_response.csv')
    
    print(f"Cell lines: {celllines.shape}")
    print(f"Expression data: {expression.shape}")
    print(f"Drug response: {drug_response.shape}")
    
    return celllines, expression, drug_response

def preprocess_expression_data(expression_df):
    """Preprocess gene expression data"""
    # Ensure Cell_Line_ID is a column, not the index
    if 'Cell_Line_ID' not in expression_df.columns:
        expression_df = expression_df.reset_index()
        expression_df = expression_df.rename(columns={'index': 'Cell_Line_ID'})
    
    # Set Cell_Line_ID as index and transpose
    expression_df = expression_df.set_index('Cell_Line_ID')
    
    # Remove any non-numeric columns before transposing
    numeric_cols = expression_df.select_dtypes(include=[np.number]).columns
    expression_df = expression_df[numeric_cols]
    
    # Transpose to have genes as columns
    expression_df = expression_df.T
    
    # Remove low variance genes
    selector = VarianceThreshold(threshold=0.01)  # Lower threshold
    expression_filtered = selector.fit_transform(expression_df)
    
    # Get selected gene names
    selected_genes = expression_df.columns[selector.get_support()]
    expression_filtered = pd.DataFrame(expression_filtered, columns=selected_genes, index=expression_df.index)
    
    # Standardize expression data
    scaler = StandardScaler()
    expression_scaled = scaler.fit_transform(expression_filtered)
    expression_scaled = pd.DataFrame(expression_scaled, columns=selected_genes, index=expression_filtered.index)
    
    print(f"Expression data after preprocessing: {expression_scaled.shape}")
    return expression_scaled, scaler

def preprocess_drug_response(drug_response_df, target_drug='5-Fluorouracil'):
    """Preprocess drug response data"""
    # Filter for target drug
    drug_data = drug_response_df[drug_response_df['Drug_Name'] == target_drug].copy()
    
    # Calculate IC50 values (handle missing values)
    drug_data['IC50'] = drug_data['IC50'].fillna(drug_data['IC50'].median())
    
    # Log transform IC50 for better distribution
    drug_data['log_IC50'] = np.log10(drug_data['IC50'])
    
    # Merge with cell line metadata
    celllines = pd.read_csv('data/gdsc_celllines.csv')
    drug_data = drug_data.merge(celllines, on='Cell_Line_ID', how='left')
    
    print(f"Drug response data for {target_drug}: {drug_data.shape}")
    return drug_data

def prepare_features_targets(expression_df, drug_response_df):
    """Prepare features (gene expression) and targets (drug response)"""
    # Ensure we have cell lines in both datasets
    if expression_df.shape[1] == 0:
        raise ValueError("Expression dataframe has no columns after preprocessing")
    
    # Find common cell lines
    common_celllines = set(expression_df.columns) & set(drug_response_df['Cell_Line_ID'])
    print(f"Common cell lines: {len(common_celllines)}")
    
    if len(common_celllines) == 0:
        # If no common cell lines, use all available data for demonstration
        print("Warning: No common cell lines found. Using all available data for demonstration.")
        # Create synthetic mapping for demonstration
        cell_ids = list(expression_df.columns)[:50]  # Use first 50 cell lines
        drug_ids = list(drug_response_df['Cell_Line_ID'])[:50]  # Use first 50 drug responses
        
        # Filter data
        X = expression_df.iloc[:, :50].T  # Transpose to get cell lines as rows
        y = drug_response_df.set_index('Cell_Line_ID').loc[drug_ids]['log_IC50']
    else:
        # Filter data to common cell lines
        X = expression_df[list(common_celllines)].T  # Transpose to get cell lines as rows
        y = drug_response_df.set_index('Cell_Line_ID').loc[list(common_celllines)]['log_IC50']
    
    # Feature selection - select top genes correlated with drug response
    # Use a smaller k if we have fewer features
    k = min(100, X.shape[1])
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected gene names
    selected_genes = X.columns[selector.get_support()]
    X_selected = pd.DataFrame(X_selected, columns=selected_genes, index=X.index)
    
    print(f"Final feature matrix: {X_selected.shape}")
    return X_selected, y, selected_genes, selector

def pathway_enrichment(selected_genes):
    """Perform pathway enrichment analysis on selected genes"""
    # Convert gene symbols to Entrez IDs (simplified for example)
    # In practice, you'd use a proper gene ID mapping service
    gene_list = selected_genes.tolist()
    
    # Perform enrichment analysis using KEGG pathways
    enr_results = gp.enrichr(
        gene_list=gene_list,
        gene_sets=['KEGG_2021_Human'],
        organism='human',
        outdir=None
    )
    
    # Get top enriched pathways
    top_pathways = enr_results.results.head(10)
    print("Top enriched pathways:")
    print(top_pathways[['Term', 'P-value', 'Adjusted P-value', 'Odds Ratio']])
    
    return enr_results