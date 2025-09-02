import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from src.data_preprocessing import load_data, preprocess_expression_data, preprocess_drug_response, prepare_features_targets

def train_models(X=None, y=None, selected_genes=None):
    """Train and evaluate regression models for drug response prediction"""
    # Load and preprocess data if not provided
    if X is None or y is None or selected_genes is None:
        celllines, expression, drug_response = load_data()
        expression_scaled, scaler = preprocess_expression_data(expression)
        drug_data = preprocess_drug_response(drug_response, target_drug='5-Fluorouracil')
        X, y, selected_genes, selector = prepare_features_targets(expression_scaled, drug_data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Initialize models
    models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'ElasticNet': ElasticNet(random_state=42),
        'SVR': SVR()
    }
    
    # Hyperparameter grids
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'ElasticNet': {
            'alpha': [0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.9]
        },
        'SVR': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
    }
    
    best_models = {}
    results = {}
    
    # Train and tune each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Scale features for SVR
        if name == 'SVR':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Best_Params': grid_search.best_params_
        }
        
        # Print results
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='r2')
        print(f"CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Find best overall model
    best_model_name = max(results, key=lambda x: results[x]['R2'])
    best_overall = best_models[best_model_name]
    
    print(f"\nBest overall model: {best_model_name}")
    print(f"Test R2: {results[best_model_name]['R2']:.4f}")
    
    # Return best model and results without saving
    # (saving will be handled by the pipeline script)
    return best_overall, results

if __name__ == "__main__":
    train_models()