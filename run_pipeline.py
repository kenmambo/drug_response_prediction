import os
import sys
import joblib
import pandas as pd
import numpy as np
from src.data_preprocessing import load_data, preprocess_expression_data, preprocess_drug_response, prepare_features_targets, pathway_enrichment
from src.model_training import train_models

def main():
    print("Starting drug response prediction pipeline...")
    
    # Step 1: Load and preprocess data
    print("\n=== Step 1: Loading and preprocessing data ===")
    celllines, expression, drug_response = load_data()
    expression_scaled, scaler = preprocess_expression_data(expression)
    drug_data = preprocess_drug_response(drug_response, target_drug='5-Fluorouracil')
    X, y, selected_genes, selector = prepare_features_targets(expression_scaled, drug_data)
    
    # Step 2: Perform pathway enrichment analysis
    print("\n=== Step 2: Performing pathway enrichment analysis ===")
    pathway_enrichment(selected_genes)
    
    # Step 3: Train models
    print("\n=== Step 3: Training models ===")
    best_model, results = train_models(X, y, selected_genes)
    
    # Step 4: Save model and preprocessing objects
    print("\n=== Step 4: Saving model and preprocessing objects ===")
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'expression_scaler.pkl')
    joblib.dump(selector, 'feature_selector.pkl')
    joblib.dump(selected_genes, 'selected_genes.pkl')
    
    # Save model results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('model_results.csv')
    
    print("\nPipeline completed successfully!")
    print("You can now run the dashboard with: streamlit run app/dashboard.py")

if __name__ == "__main__":
    main()