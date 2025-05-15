import pandas as pd
import numpy as np
import io

def get_sample_data():
    """
    Load the sea level dataset from the CSV file.
    
    Returns:
    pandas.DataFrame: Dataset with climate and sea level data
    """
    try:
        # Load sea level data from CSV file
        df = pd.read_csv('sea_level_data.csv')
        
        # Rename column for compatibility with existing app
        if 'SeaLevelRise' in df.columns and 'Sea_Level_mm' not in df.columns:
            df = df.rename(columns={'SeaLevelRise': 'Sea_Level_mm'})
            
        # Convert values to millimeters if they're in meters
        if df['Sea_Level_mm'].max() < 10:  # likely in meters
            df['Sea_Level_mm'] = df['Sea_Level_mm'] * 1000  # convert to mm
            
        # Only keep Year and Sea_Level_mm columns
        if 'Year' in df.columns and 'Sea_Level_mm' in df.columns:
            # Round Year to nearest integer for better readability
            df['Year'] = df['Year'].round().astype(int)
            # Keep only these two columns
            return df[['Year', 'Sea_Level_mm']]
        else:
            raise ValueError("Required columns not found in dataset")
            
    except Exception as e:
        print(f"Error loading sea level data: {str(e)}")
        # Create a minimal fallback dataset in case of error
        years = list(range(1970, 2023))
        sea_levels = [i * 3.5 for i in range(len(years))]
        return pd.DataFrame({
            'Year': years, 
            'Sea_Level_mm': sea_levels
        })
