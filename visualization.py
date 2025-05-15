import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_actual_vs_predicted(years, y_true, y_pred, model_name):
    """
    Plot actual vs predicted sea level.
    
    Parameters:
    years (numpy.ndarray): Array of years corresponding to test data
    y_true (pandas.Series): Actual sea level values
    y_pred (numpy.ndarray): Predicted sea level values
    model_name (str): Name of the model used for prediction
    
    Returns:
    matplotlib.figure.Figure: The created figure
    """
    # Set the style
    sns.set_style("whitegrid")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot actual values
    ax.scatter(years, y_true, color='blue', alpha=0.7, label='Actual Sea Level')
    
    # Plot predicted values
    ax.scatter(years, y_pred, color='red', alpha=0.7, label=f'{model_name} Predictions')
    
    # Add a line to connect the predicted points for better visualization
    sorted_indices = np.argsort(years)
    ax.plot(years[sorted_indices], y_pred[sorted_indices], color='red', linestyle='--', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel('Sea Level (mm)')
    ax.set_title(f'Actual vs Predicted Sea Level ({model_name})')
    
    # Add legend
    ax.legend()
    
    # Improve layout
    plt.tight_layout()
    
    return fig

def plot_model_comparison(metrics):
    """
    Plot a comparison of model performance metrics.
    
    Parameters:
    metrics (dict): Dictionary with model evaluation metrics
    
    Returns:
    matplotlib.figure.Figure: The created figure
    """
    # Set the style
    sns.set_style("whitegrid")
    
    # Create figure and axis for bar plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create a DataFrame from the metrics dictionary
    metrics_df = pd.DataFrame(metrics)
    
    # Plot MAE
    sns.barplot(x='Model', y='MAE', data=metrics_df, ax=axes[0], palette='Blues_d')
    axes[0].set_title('Mean Absolute Error (Lower is better)')
    axes[0].set_ylabel('MAE')
    
    # Plot MSE
    sns.barplot(x='Model', y='MSE', data=metrics_df, ax=axes[1], palette='Reds_d')
    axes[1].set_title('Mean Squared Error (Lower is better)')
    axes[1].set_ylabel('MSE')
    
    # Plot R²
    sns.barplot(x='Model', y='R²', data=metrics_df, ax=axes[2], palette='Greens_d')
    axes[2].set_title('R² Score (Higher is better)')
    axes[2].set_ylabel('R²')
    
    # Improve layout
    plt.tight_layout()
    
    return fig

def plot_sea_level_prediction(historical_data, prediction_data, model_name):
    """
    Plot historical sea level data with future predictions.
    
    Parameters:
    historical_data (pandas.DataFrame): DataFrame with Year and Sea_Level_mm columns
    prediction_data (pandas.DataFrame): DataFrame with Year and Predicted_Sea_Level_mm columns
    model_name (str): Name of the model used for prediction
    
    Returns:
    matplotlib.figure.Figure: The created figure
    """
    # Set the style
    sns.set_style("whitegrid")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data
    ax.plot(historical_data["Year"], historical_data["Sea_Level_mm"], 
            marker='o', linestyle='-', color='blue', label='Historical Data')
    
    # Plot predicted data
    ax.plot(prediction_data["Year"], prediction_data["Predicted_Sea_Level_mm"], 
            marker='x', linestyle='--', color='red', label=f'{model_name} Prediction')
    
    # Fill the area between current and future
    last_historical_year = historical_data["Year"].max()
    ax.axvline(x=last_historical_year, color='gray', linestyle=':', alpha=0.7)
    
    # Add a text annotation
    ax.text(last_historical_year + 2, min(historical_data["Sea_Level_mm"].min(), 
            prediction_data["Predicted_Sea_Level_mm"].min()), 
            'Future Predictions', rotation=90, verticalalignment='bottom', alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel('Sea Level (mm)')
    ax.set_title(f'Sea Level Rise Prediction ({model_name})')
    
    # Add legend
    ax.legend()
    
    # Improve layout
    plt.tight_layout()
    
    return fig
