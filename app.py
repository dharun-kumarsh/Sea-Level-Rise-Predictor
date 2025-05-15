import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO

from data_processing import load_data, preprocess_data, split_data
from model_training import train_models, evaluate_models, save_models, load_models
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

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["📊 Data & Training", "🔮 Prediction", "📈 Visualization"])

with tab1:
    st.header("Data Loading and Model Training")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Data loading section
        st.subheader("Load Dataset")
        data_option = st.radio(
            "Select data source:",
            ["Use sample data", "Upload your own CSV file"]
        )
        
        if data_option == "Use sample data":
            # Load sample data
            df = get_sample_data()
            st.success("Sample data loaded successfully!")
        else:
            # File uploader
            uploaded_file = st.file_uploader("Upload CSV file with Year and Sea_Level_mm columns", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    # Read the CSV file
                    df = pd.read_csv(uploaded_file)
                    
                    # Validate the required columns exist
                    required_columns = ["Year", "Sea_Level_mm"]
                    if not all(col in df.columns for col in required_columns):
                        st.error(f"The CSV file must contain these columns: {', '.join(required_columns)}")
                        df = None
                    else:
                        st.success("Data loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
                    df = None
            else:
                df = None
                
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
            train_test_split = st.slider("Train-Test Split Ratio", 0.1, 0.9, 0.8, 0.05, 
                                        help="Percentage of data to use for training")
            
            svr_c = st.number_input("SVR C parameter", 0.1, 1000.0, 100.0, 
                                   help="Regularization parameter for SVR model")
            svr_epsilon = st.number_input("SVR Epsilon", 0.01, 1.0, 0.1, 0.01, 
                                         help="Epsilon value for SVR model")
            
            rf_n_estimators = st.number_input("Random Forest n_estimators", 10, 500, 100, 10, 
                                            help="Number of trees in the Random Forest")
            rf_max_depth = st.number_input("Random Forest max_depth", 1, 30, 10, 1, 
                                         help="Maximum depth of the trees")
            
            # Training button
            if st.button("Train Models"):
                with st.spinner("Training models..."):
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
                    
                    # Train the models
                    svr_params = {'C': svr_c, 'epsilon': svr_epsilon}
                    rf_params = {'n_estimators': rf_n_estimators, 'max_depth': rf_max_depth}
                    
                    models = train_models(X_train, y_train, svr_params, rf_params)
                    
                    # Evaluate the models
                    metrics = evaluate_models(models, X_test, y_test)
                    
                    # Store models and metrics in session state
                    st.session_state.models = models
                    st.session_state.metrics = metrics
                    
                    # Make predictions on test data for visualization
                    st.session_state.predictions = {
                        'SVR': models['SVR'].predict(X_test),
                        'Random Forest': models['RF'].predict(X_test)
                    }
                    
                    st.success("Models trained successfully!")
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
            # Model selection
            model_choice = st.selectbox(
                "Select the model for prediction:",
                ["SVR", "Random Forest"],
                index=0
            )
            
            # Year selection for prediction
            if st.session_state.data is not None:
                min_year = int(st.session_state.data["Year"].max()) + 1
                max_year = min_year + 100  # Allow prediction up to 100 years in the future
            else:
                min_year = 2023
                max_year = 2123
            
            # User can either enter a specific year or select a range
            pred_year_option = st.radio(
                "Select prediction type:",
                ["Single year", "Year range"]
            )
            
            if pred_year_option == "Single year":
                prediction_year = st.number_input(
                    "Enter the year to predict:",
                    min_value=min_year,
                    max_value=max_year,
                    value=2050
                )
                years_to_predict = [prediction_year]
            else:
                start_year = st.number_input(
                    "Start year:",
                    min_value=min_year,
                    max_value=max_year-1,
                    value=2030
                )
                end_year = st.number_input(
                    "End year:",
                    min_value=start_year+1,
                    max_value=max_year,
                    value=min(start_year+50, max_year)
                )
                years_to_predict = list(range(start_year, end_year+1))
            
            # Button to make prediction
            if st.button("Predict Sea Level"):
                with st.spinner("Making predictions..."):
                    # Get the selected model
                    selected_model = st.session_state.models["SVR"] if model_choice == "SVR" else st.session_state.models["RF"]
                    
                    # Make the prediction
                    predictions = predict_sea_level(selected_model, years_to_predict, st.session_state.scaler)
                    
                    # Store predictions for visualization
                    st.session_state.future_predictions = {
                        'Years': years_to_predict,
                        'Predictions': predictions,
                        'Model': model_choice
                    }
                    
                    st.success(f"Prediction completed for {model_choice} model!")
        
        with col2:
            if 'future_predictions' in st.session_state:
                st.subheader("Prediction Results")
                
                if len(st.session_state.future_predictions['Years']) == 1:
                    year = st.session_state.future_predictions['Years'][0]
                    prediction = st.session_state.future_predictions['Predictions'][0]
                    model = st.session_state.future_predictions['Model']
                    
                    st.markdown(f"""
                    ### Predicted Sea Level for {year}
                    
                    Using the **{model}** model:
                    
                    **{prediction:.2f} mm** above the reference level
                    """)
                    
                    # Calculate the change from the latest recorded year
                    latest_year = st.session_state.data["Year"].max()
                    latest_level = st.session_state.data.loc[st.session_state.data["Year"] == latest_year, "Sea_Level_mm"].values[0]
                    change = prediction - latest_level
                    
                    st.markdown(f"""
                    **Change from {latest_year}**: {change:.2f} mm ({'+' if change > 0 else ''}{change/latest_level*100:.2f}%)
                    """)
                else:
                    # Create a dataframe with predictions for multiple years
                    pred_df = pd.DataFrame({
                        'Year': st.session_state.future_predictions['Years'],
                        'Predicted_Sea_Level_mm': st.session_state.future_predictions['Predictions']
                    })
                    
                    st.write(f"Predictions using {st.session_state.future_predictions['Model']} model:")
                    st.write(pred_df)
                    
                    # Plot the predictions
                    fig = plot_sea_level_prediction(
                        st.session_state.data,
                        pred_df,
                        model_name=st.session_state.future_predictions['Model']
                    )
                    st.pyplot(fig)

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
            # Allow user to select which model to visualize
            model_for_viz = st.selectbox(
                "Select model to visualize:",
                ["SVR", "Random Forest"]
            )
            
            # Create the actual vs predicted plot
            if 'test_years' in st.session_state and st.session_state.test_years is not None:
                fig = plot_actual_vs_predicted(
                    st.session_state.test_years,  
                    st.session_state.y_test, 
                    st.session_state.predictions[model_for_viz if model_for_viz == "SVR" else "Random Forest"],
                    model_name=model_for_viz
                )
                st.pyplot(fig)
            else:
                st.warning("Please train models first before visualizing results.")
            
        else:  # Model Comparison
            # Create the model comparison plot
            fig = plot_model_comparison(st.session_state.metrics)
            st.pyplot(fig)
            
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
