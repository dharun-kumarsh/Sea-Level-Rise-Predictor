import pandas as pd
import numpy as np
import io

def get_sample_data():
    """
    Generate a sample dataset for sea level rise.
    This function creates a realistic sample dataset based on historical measurements.
    
    Returns:
    pandas.DataFrame: Sample dataset with Year and Sea_Level_mm columns
    """
    # Create a sample dataset with realistic sea level rise
    # Data is loosely based on satellite measurements since 1993
    csv_data = """Year,Sea_Level_mm
1993,0
1994,2.1
1995,4.2
1996,6.8
1997,9.1
1998,12.3
1999,15.5
2000,17.3
2001,20.2
2002,22.9
2003,25.4
2004,27.9
2005,30.1
2006,32.5
2007,34.7
2008,37.8
2009,40.3
2010,43.6
2011,45.9
2012,49.3
2013,52.7
2014,57.1
2015,62.6
2016,68.2
2017,73.4
2018,78.9
2019,84.3
2020,90.1
2021,96.2
2022,102.5"""
    
    # Load the CSV string into a DataFrame
    df = pd.read_csv(io.StringIO(csv_data))
    
    return df
