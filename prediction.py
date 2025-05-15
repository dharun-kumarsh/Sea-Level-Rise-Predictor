import numpy as np
import pandas as pd

def predict_sea_level(model, years, scaler):
    """
    Predict sea level for given years using a trained model.
    
    Parameters:
    model: Trained machine learning model
    years (list): List of years to predict
    scaler: The scaler used to transform the original data
    
    Returns:
    numpy.ndarray: Array of predicted sea level values
    """
    # Convert years to a numpy array
    years_array = np.array(years).reshape(-1, 1)
    
    # Scale the years
    years_scaled = scaler.transform(years_array)
    
    # Make predictions
    predictions = model.predict(years_scaled)
    
    return predictions

def generate_prediction_dataframe(years, predictions):
    """
    Generate a DataFrame with prediction results.
    
    Parameters:
    years (list): List of years
    predictions (numpy.ndarray): Array of predicted sea level values
    
    Returns:
    pandas.DataFrame: DataFrame with Year and Predicted_Sea_Level_mm columns
    """
    # Create a DataFrame
    df = pd.DataFrame({
        'Year': years,
        'Predicted_Sea_Level_mm': predictions
    })
    
    return df
