import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def train_models(X_train, y_train, svr_params, rf_params):
    """
    Train SVR and Random Forest models on the given data.
    
    Parameters:
    X_train (numpy.ndarray): Training features
    y_train (pandas.Series): Training target
    svr_params (dict): Parameters for SVR model
    rf_params (dict): Parameters for Random Forest model
    
    Returns:
    dict: Trained models
    """
    # Initialize models with parameters
    svr = SVR(
        C=svr_params.get('C', 100.0),
        epsilon=svr_params.get('epsilon', 0.1),
        kernel='rbf',
        gamma='scale'
    )
    
    rf = RandomForestRegressor(
        n_estimators=rf_params.get('n_estimators', 100),
        max_depth=rf_params.get('max_depth', 10),
        random_state=42
    )
    
    # Train models
    svr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    
    # Return trained models
    return {
        'SVR': svr,
        'RF': rf
    }

def evaluate_models(models, X_test, y_test):
    """
    Evaluate the trained models on test data.
    
    Parameters:
    models (dict): Trained models
    X_test (numpy.ndarray): Test features
    y_test (pandas.Series): Test target
    
    Returns:
    dict: Dictionary with model evaluation metrics
    """
    metrics = {
        'Model': [],
        'MAE': [],
        'MSE': [],
        'R²': []
    }
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Add metrics to the dictionary
        metrics['Model'].append(name)
        metrics['MAE'].append(mae)
        metrics['MSE'].append(mse)
        metrics['R²'].append(r2)
    
    return metrics

def save_models(models, directory='models'):
    """
    Save trained models to disk.
    
    Parameters:
    models (dict): Trained models
    directory (str): Directory to save models
    
    Returns:
    bool: True if saved successfully
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Save each model
    for name, model in models.items():
        file_path = os.path.join(directory, f"{name}_model.joblib")
        joblib.dump(model, file_path)
    
    return True

def load_models(directory='models'):
    """
    Load trained models from disk.
    
    Parameters:
    directory (str): Directory where models are saved
    
    Returns:
    dict: Loaded models
    """
    models = {}
    
    # Check if directory exists
    if not os.path.exists(directory):
        return models
    
    # Load SVR model if it exists
    svr_path = os.path.join(directory, "SVR_model.joblib")
    if os.path.exists(svr_path):
        models['SVR'] = joblib.load(svr_path)
    
    # Load RF model if it exists
    rf_path = os.path.join(directory, "RF_model.joblib")
    if os.path.exists(rf_path):
        models['RF'] = joblib.load(rf_path)
    
    return models
