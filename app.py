import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from data_processing import load_data, preprocess_data, split_data
from model_training import evaluate_models, save_models, load_models
from visualization import plot_actual_vs_predicted, plot_model_comparison, plot_sea_level_prediction
from prediction import predict_sea_level
from utils import get_sample_data

# Set page configuration
st.set_page_config(
    page_title="Sea Level Rise Prediction",
    page_icon="🌊",
    layout="wide"
)

# Title and description
st.title("🌊 Sea Level Rise Prediction")
st.markdown("""
This application predicts future sea level rise using machine learning models:
* Support Vector Regressor (SVR)
* Random Forest Regressor
""")

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'test_years' not in st.session_state:
    st.session_state.test_years = None
if 'future_predictions' not in st.session_state:
    st.session_state.future_predictions = None

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["📊 Data & Training", "🔮 Prediction", "📈 Visualization"])

with tab1:
    st.header("Data Loading and Model Training")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Data loading section
        st.subheader("Dataset")
        # Only use sample data
        df = get_sample_data()
        st.success("Sample data loaded successfully!")
                
        # Display the dataframe if it's loaded
        if df is not None:
            st.session_state.data = df
            st.write("Preview of the dataset:")
            st.write(df.head())
            
            # Display some statistics
            st.write("Data Statistics:")
            st.write(df.describe())
            
            # Check for missing values
            missing_values = df.isnull().sum().sum()
            if missing_values > 0:
                st.warning(f"Dataset contains {missing_values} missing values. They will be handled during preprocessing.")
            else:
                st.info("No missing values found in the dataset.")
                
    with col2:
        # Model training section
        st.subheader("Model Training")
        
        if st.session_state.data is not None:
            # Model selection
            model_choice = st.radio(
                "Select model to train:",
                ["SVR", "Random Forest"]
            )
            
            # Fixed parameters
            train_test_split = 0.8
            svr_params = {'C': 100.0, 'epsilon': 0.1}
            rf_params = {'n_estimators': 100, 'max_depth': 10}
            
            # Training button
            if st.button("Train Selected Model"):
                with st.spinner(f"Training {model_choice} model..."):
                    # Preprocess the data
                    X, y, scaler = preprocess_data(st.session_state.data)
                    
                    # Split the data
                    X_train, X_test, y_train, y_test = split_data(X, y, train_size=train_test_split)
                    
                    # Save splits to session state for later use
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.scaler = scaler
                    
                    # Save the original years corresponding to the test set
                    # We need to track which rows go into the test set
                    n_samples = len(X)
                    indices = np.arange(n_samples)
                    n_train = int(train_test_split * n_samples)
                    # Use same random state as in the split_data function
                    np.random.seed(42)
                    # Shuffle indices
                    np.random.shuffle(indices)
                    # Get test indices
                    test_indices = indices[n_train:]
                    # Save the test years
                    st.session_state.test_years = st.session_state.data.iloc[test_indices]["Year"].values
                    
                    # Train only the selected model
                    if model_choice == "SVR":
                        svr = SVR(C=svr_params['C'], epsilon=svr_params['epsilon'], kernel='rbf', gamma='scale')
                        svr.fit(X_train, y_train)
                        models = {'SVR': svr}
                        model_predictions = {'SVR': svr.predict(X_test)}
                    else:  # Random Forest
                        rf = RandomForestRegressor(n_estimators=rf_params['n_estimators'], max_depth=rf_params['max_depth'], random_state=42)
                        rf.fit(X_train, y_train)
                        models = {'RF': rf}
                        model_predictions = {'Random Forest': rf.predict(X_test)}
                    
                    # Evaluate the model
                    metrics = evaluate_models(models, X_test, y_test)
                    
                    # Store models and metrics in session state
                    st.session_state.models = models
                    st.session_state.metrics = metrics
                    
                    # Make predictions on test data for visualization
                    st.session_state.predictions = model_predictions
                    
                    st.success(f"{model_choice} model trained successfully!")
        else:
            st.info("Please load a dataset first to train models.")
    
    # Display model metrics if available
    if st.session_state.metrics is not None:
        st.subheader("Model Performance Metrics")
        metrics_df = pd.DataFrame(st.session_state.metrics)
        st.write(metrics_df)

with tab2:
    st.header("Sea Level Prediction")
    
    if st.session_state.models is None:
        st.warning("Please train models in the 'Data & Training' tab first.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Get the model that was trained
            # Determine the available model in session state
            available_model = None
            if 'models' in st.session_state and st.session_state.models:
                if 'SVR' in st.session_state.models:
                    available_model = "SVR"
                elif 'RF' in st.session_state.models:
                    available_model = "Random Forest"
            
            if available_model:
                st.write(f"Using trained model: **{available_model}**")
            else:
                st.warning("No trained model found. Please train a model first in the 'Data & Training' tab.")
            
            # Year selection for prediction
            # Check if data exists in session state
            if hasattr(st.session_state, 'data') and st.session_state.data is not None:
                # Use data from session state
                min_year = int(st.session_state.data["Year"].max()) + 1
                max_year = min_year + 100  # Allow prediction up to 100 years in the future
            else:
                # Default values if no data is available
                min_year = 2023
                max_year = 2123
            
            # Simplified prediction - just a single year selection with a slider
            prediction_year = st.slider(
                "Select year to predict:",
                min_value=min_year,
                max_value=max_year,
                value=2050,
                step=1
            )
            years_to_predict = [prediction_year]
            
            # Button to make prediction
            if available_model and st.button("Predict Sea Level"):
                with st.spinner("Making predictions..."):
                    # Get the trained model
                    if available_model == "SVR":
                        selected_model = st.session_state.models["SVR"]
                    else:  # Random Forest
                        selected_model = st.session_state.models["RF"]
                    
                    # Make the prediction
                    predictions = predict_sea_level(selected_model, years_to_predict, st.session_state.scaler)
                    
                    # Store predictions for visualization
                    st.session_state.future_predictions = {
                        'Years': years_to_predict,
                        'Predictions': predictions,
                        'Model': available_model
                    }
                    
                    st.success(f"Prediction completed for {available_model} model!")
        
        with col2:
            if hasattr(st.session_state, 'future_predictions') and st.session_state.future_predictions is not None:
                st.subheader("Prediction Results")
                
                # Access prediction data safely
                years = st.session_state.future_predictions.get('Years', [])
                predictions = st.session_state.future_predictions.get('Predictions', [])
                model = st.session_state.future_predictions.get('Model', 'Unknown')
                
                if len(years) == 1:
                    year = years[0]
                    prediction = predictions[0]
                    
                    st.markdown(f"""
                    ### Predicted Sea Level for {year}
                    
                    Using the **{model}** model:
                    
                    **{prediction:.2f} mm** above the reference level
                    """)
                    
                    # Calculate the change from the latest recorded year
                    if hasattr(st.session_state, 'data') and st.session_state.data is not None:
                        latest_year = st.session_state.data["Year"].max()
                        latest_level = st.session_state.data.loc[st.session_state.data["Year"] == latest_year, "Sea_Level_mm"].values[0]
                    else:
                        latest_year = 2022  # Fallback to most recent year in sample data
                        latest_level = 102.5  # Fallback to most recent level in sample data
                    change = prediction - latest_level
                    
                    st.markdown(f"""
                    **Change from {latest_year}**: {change:.2f} mm ({'+' if change > 0 else ''}{change/latest_level*100:.2f}%)
                    """)
                else:
                    # Create a dataframe with predictions for multiple years
                    pred_df = pd.DataFrame({
                        'Year': years,
                        'Predicted_Sea_Level_mm': predictions
                    })
                    
                    st.write(f"Predictions using {model} model:")
                    st.write(pred_df)
                    
                    # Plot the predictions safely
                    if hasattr(st.session_state, 'data') and st.session_state.data is not None:
                        fig = plot_sea_level_prediction(
                            st.session_state.data,
                            pred_df,
                            model_name=model
                        )
                        st.pyplot(fig)
                    else:
                        st.warning("Cannot plot predictions without historical data.")

with tab3:
    st.header("Visualization")
    
    if st.session_state.models is None:
        st.warning("Please train models in the 'Data & Training' tab first.")
    else:
        viz_option = st.radio(
            "Select visualization type:",
            ["Actual vs Predicted", "Model Comparison"]
        )
        
        if viz_option == "Actual vs Predicted":
            # Determine the available model in session state
            available_model = None
            if 'models' in st.session_state and st.session_state.models:
                if 'SVR' in st.session_state.models:
                    available_model = "SVR"
                elif 'RF' in st.session_state.models:
                    available_model = "Random Forest"
            
            if available_model:
                st.write(f"Showing results for trained model: **{available_model}**")
                
                # Create the actual vs predicted plot if test data is available
                if 'test_years' in st.session_state and st.session_state.test_years is not None:
                    # Get the correct predictions key based on the model
                    prediction_key = available_model if available_model == "SVR" else "Random Forest"
                    
                    if prediction_key in st.session_state.predictions:
                        fig = plot_actual_vs_predicted(
                            st.session_state.test_years,  
                            st.session_state.y_test, 
                            st.session_state.predictions[prediction_key],
                            model_name=available_model
                        )
                        st.pyplot(fig)
                    else:
                        st.warning(f"No predictions found for {available_model}.")
                else:
                    st.warning("Please train the model first before visualizing results.")
            else:
                st.warning("No trained model found. Please train a model in the 'Data & Training' tab.")
            
        else:  # Model Comparison
            # Create the model comparison plot
            if hasattr(st.session_state, 'metrics') and st.session_state.metrics is not None:
                fig = plot_model_comparison(st.session_state.metrics)
                st.pyplot(fig)
            else:
                st.warning("Please train models first to view model comparison.")
            
            # Additional information about the metrics
            st.subheader("Understanding the Metrics")
            st.markdown("""
            - **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual values. Lower is better.
            - **Mean Squared Error (MSE)**: Average squared difference between predicted and actual values. Lower is better.
            - **R² Score**: Proportion of variance in the dependent variable that is predictable from the independent variable(s). Closer to 1 is better.
            """)

# Add a footer
st.markdown("---")
st.markdown("Sea Level Rise Prediction App | Created with Streamlit")
