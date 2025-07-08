import streamlit as st
import pandas as pd
import numpy as np
from model import FraudDetectionModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import io
from PIL import Image
import os
from typing import Optional, Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

@st.cache_resource
def load_model() -> Optional[FraudDetectionModel]:
    """
    Cache the model loading to prevent reloading on every interaction.
    
    Returns:
        Optional[FraudDetectionModel]: Loaded model or None if loading fails
    """
    try:
        model = FraudDetectionModel()
        model_path = 'bilstm_fraud_detection.h5'
        scaler_path = 'scaler.npy'
        
        # Check if model files exist
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None
        if not os.path.exists(scaler_path):
            st.error(f"Scaler file not found at {scaler_path}")
            return None
            
        if model.load_model():
            st.success("Model loaded successfully!")
            return model
        return None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Error loading model: {str(e)}")
        return None

def process_data_in_batches(df: pd.DataFrame, batch_size: int = 1000) -> pd.DataFrame:
    """
    Process data in batches to prevent memory issues.
    
    Args:
        df: Input DataFrame
        batch_size: Size of each batch
        
    Yields:
        DataFrame: Batch of data
    """
    total_rows = len(df)
    for i in range(0, total_rows, batch_size):
        yield df.iloc[i:min(i + batch_size, total_rows)]

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        plt.Figure: Confusion matrix plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

def display_model_info() -> None:
    """Display model information in the sidebar."""
    with st.sidebar:
        st.title("Model Information")
        st.markdown("""
        ### Model Architecture
        - BiLSTM with 64 and 32 units
        - Dropout Regularization
        - Dense layers with ReLU activation
        - Sigmoid Output
        
        ### Performance Metrics
        Accuracy: 0.999663
        Precision: 0.918367
        Recall: 0.999831
        F1-Score: 0.955471
        """)
        
        try:
            st.image("model_architecture.png", caption="BiLSTM Model Architecture", use_column_width=True)
        except Exception as e:
            logger.warning(f"Model architecture image not found: {str(e)}")
            st.warning("Model architecture image not found")

def process_uploaded_file(uploaded_file: Any) -> Optional[pd.DataFrame]:
    """
    Process the uploaded CSV file.
    
    Args:
        uploaded_file: Uploaded file object
        
    Returns:
        Optional[pd.DataFrame]: Processed DataFrame or None if processing fails
    """
    try:
        with st.spinner("Loading data..."):
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            return df
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        st.error(f"Error processing file: {str(e)}")
        return None

def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the input data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    expected_features = 30  # V1-V28, Time, Amount
    if len(df.columns) != expected_features and 'Class' not in df.columns:
        st.error(f"Input data must have exactly {expected_features} features (V1-V28, Time, Amount)")
        return False
    return True

def display_results(results_df: pd.DataFrame, y_true: Optional[np.ndarray] = None) -> None:
    """
    Display results in tabs.
    
    Args:
        results_df: DataFrame containing results
        y_true: Optional true labels for evaluation
    """
    tab1, tab2, tab3 = st.tabs(["Results", "Evaluation", "Download"])
    
    with tab1:
        st.subheader("Detection Results")
        st.dataframe(results_df)
        
        # Add a pie chart of fraud vs non-fraud predictions
        fraud_counts = results_df['Fraud_Prediction'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(fraud_counts, labels=['Normal', 'Fraud'], autopct='%1.1f%%')
        ax.set_title('Distribution of Predictions')
        st.pyplot(fig)
    
    with tab2:
        if y_true is not None:
            st.subheader("Model Evaluation")
            
            # Plot confusion matrix
            fig = plot_confusion_matrix(y_true, results_df['Fraud_Prediction'])
            st.pyplot(fig)
            
            # Display classification report
            report = classification_report(y_true, results_df['Fraud_Prediction'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
        else:
            st.info("No ground truth labels available for evaluation")
    
    with tab3:
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results",
            data=csv,
            file_name="fraud_detection_results.csv",
            mime="text/csv"
        )

def main() -> None:
    """Main function to run the Streamlit app."""
    # Display model information
    display_model_info()

    # Main content
    st.title("Credit Card Fraud Detection System")
    st.write("Upload a CSV file containing credit card transaction data for fraud detection.")
    
    # Initialize model if not already loaded
    if st.session_state.model is None:
        with st.spinner("Loading model..."):
            st.session_state.model = load_model()
            if st.session_state.model is None:
                st.error("""
                Model not found. Please ensure the following files are present in your repository:
                - bilstm_fraud_detection.h5
                - scaler.npy
                
                If you're deploying on Streamlit Cloud, make sure these files are included in your GitHub repository.
                """)
                return
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Process uploaded file
        df = process_uploaded_file(uploaded_file)
        if df is None:
            return
            
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Validate data
        if not validate_data(df):
            return
        
        # Remove Class column if it exists
        y_true = df['Class'] if 'Class' in df.columns else None
        if 'Class' in df.columns:
            df = df.drop('Class', axis=1)
        
        # Create two columns for threshold and visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Threshold slider
            threshold = st.slider("Fraud Detection Threshold", 0.0, 1.0, 0.5, 0.01)
        
        with col2:
            st.write("Real-time Fraud Detection Visualization")
        
        # Make predictions
        if st.button("Detect Fraud"):
            with st.spinner("Processing..."):
                try:
                    # Process data in batches
                    all_predictions = []
                    progress_bar = st.progress(0)
                    
                    for i, batch_df in enumerate(process_data_in_batches(df)):
                        batch_predictions = st.session_state.model.predict(batch_df.values)
                        all_predictions.extend(batch_predictions)
                        progress_bar.progress((i + 1) * 1000 / len(df))
                    
                    predictions_prob = np.array(all_predictions)
                    predictions = (predictions_prob > threshold).astype(int)
                    st.session_state.predictions = predictions
                    
                    # Create results dataframe
                    results_df = df.copy()
                    results_df['Fraud_Probability'] = predictions_prob
                    results_df['Fraud_Prediction'] = predictions
                    
                    # Display results
                    display_results(results_df, y_true)
                    
                except Exception as e:
                    logger.error(f"Error during prediction: {str(e)}")
                    st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main() 
