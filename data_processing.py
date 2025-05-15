import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load the sea level dataset from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pandas.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        # Validate the required columns
        required_columns = ["Year", "Sea_Level_mm"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"The CSV file must contain these columns: {', '.join(required_columns)}")
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def preprocess_data(df):
    """
    Preprocess the sea level dataset.
    
    Parameters:
    df (pandas.DataFrame): The dataset with Year and Sea_Level_mm columns
    
    Returns:
    tuple: (X features, y target, scaler)
    """
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        # Handle missing values by interpolation
        df = df.interpolate(method='linear')
        # If there are still missing values (at the beginning), fill with the first valid value
        df = df.fillna(method='bfill')
        # If there are still missing values (at the end), fill with the last valid value
        df = df.fillna(method='ffill')
    
    # Extract features and target
    X = df[["Year"]].copy()
    y = df["Sea_Level_mm"].copy()
    
    # Scale the features (important especially for SVR)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def split_data(X, y, train_size=0.8, random_state=42):
    """
    Split the data into training and testing sets.
    
    Parameters:
    X (numpy.ndarray): Scaled feature matrix
    y (pandas.Series): Target variable
    train_size (float): Proportion of data to use for training (default 0.8)
    random_state (int): Random seed for reproducibility
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, train_size=train_size, random_state=random_state)
