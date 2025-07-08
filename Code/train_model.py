import pandas as pd
import numpy as np
from model import FraudDetectionModel
from sklearn.model_selection import train_test_split
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str = 'creditcard.csv') -> pd.DataFrame:
    """
    Load the credit card fraud dataset.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the dataset
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def main():
    """Main function to train and save the model."""
    try:
        # Initialize model
        model = FraudDetectionModel()
        
        # Load data
        df = load_data()
        
        # Preprocess data
        X, y = model.preprocess_data(df)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Balance training data
        X_train_balanced, y_train_balanced = model.balance_data(X_train, y_train)
        
        # Build model
        input_shape = (1, X.shape[2])  # (time_steps, features)
        model.build_model(input_shape)
        
        # Train model
        logger.info("Starting model training...")
        history = model.train(
            X_train_balanced, y_train_balanced,
            X_val, y_val,
            epochs=50,
            batch_size=32
        )
        
        # Save model and scaler
        model.save_model()
        
        logger.info("Training completed successfully!")
        logger.info("Model and scaler files have been saved.")
        
    except Exception as e:
        logger.error(f"Error in training process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 