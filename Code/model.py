import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
from typing import Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionModel:
    def __init__(self):
        """Initialize the FraudDetectionModel with a scaler and empty model."""
        self.scaler = StandardScaler()
        self.model = None
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the input dataframe.
        
        Args:
            df: Input DataFrame containing features and target
            
        Returns:
            Tuple of (X_reshaped, y) where X_reshaped is the preprocessed features
            and y is the target variable
        """
        try:
            # Drop duplicates and null values
            df = df.drop_duplicates()
            df = df.dropna()
            
            if 'Class' not in df.columns:
                raise ValueError("Target column 'Class' not found in the dataset")
            
            # Separate features and target
            X = df.drop('Class', axis=1)
            y = df['Class']
            
            # Scale the features
            X_scaled = self.scaler.fit_transform(X)
            
            # Reshape for BiLSTM (samples, time_steps, features)
            X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
            
            return X_reshaped, y
        except Exception as e:
            logger.error(f"Error in preprocessing data: {str(e)}")
            raise
    
    def balance_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to balance the dataset.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            Tuple of balanced (X, y)
        """
        try:
            smote = SMOTE(random_state=42)
            X_reshaped = X.reshape(X.shape[0], X.shape[2])
            X_balanced, y_balanced = smote.fit_resample(X_reshaped, y)
            return X_balanced.reshape(X_balanced.shape[0], 1, X_balanced.shape[1]), y_balanced
        except Exception as e:
            logger.error(f"Error in balancing data: {str(e)}")
            raise
    
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build the BiLSTM model.
        
        Args:
            input_shape: Shape of input data (time_steps, features)
            
        Returns:
            Compiled Keras model
        """
        try:
            model = Sequential([
                Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
                Dropout(0.3),
                Bidirectional(LSTM(32)),
                Dropout(0.3),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 
                        tf.keras.metrics.Precision(),
                        tf.keras.metrics.Recall(),
                        tf.keras.metrics.AUC()]
            )
            
            self.model = model
            return model
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray, 
              epochs: int = 50, batch_size: int = 32) -> tf.keras.callbacks.History:
        """
        Train the model with callbacks.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        try:
            if self.model is None:
                raise ValueError("Model not built. Call build_model() first.")
                
            callbacks = [
                ModelCheckpoint(
                    'bilstm_fraud_detection.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min'
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            return history
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Array of predictions
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded")
            
            # Ensure input is properly shaped
            if len(X.shape) == 2:
                X = X.reshape(X.shape[0], 1, X.shape[1])
            
            # Scale the input
            X_scaled = self.scaler.transform(X.reshape(X.shape[0], X.shape[2]))
            X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
            
            return self.model.predict(X_reshaped)
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def save_model(self, path: str = 'bilstm_fraud_detection.h5') -> None:
        """
        Save the model and scaler.
        
        Args:
            path: Path to save the model
        """
        try:
            if self.model is None:
                raise ValueError("No model to save")
                
            self.model.save(path, save_format='h5')
            np.save('scaler.npy', self.scaler)
            logger.info(f"Model saved successfully at {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path: str = 'bilstm_fraud_detection.h5') -> bool:
        """
        Load the model and scaler.
        
        Args:
            path: Path to the saved model
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            if not os.path.exists(path):
                logger.error(f"Model file not found at {path}")
                return False
                
            # Try loading with custom_objects to handle any custom layers
            custom_objects = {
                'Bidirectional': Bidirectional,
                'LSTM': LSTM,
                'Dense': Dense,
                'Dropout': Dropout
            }
            self.model = load_model(path, custom_objects=custom_objects)
            
            # Load the scaler
            scaler_path = 'scaler.npy'
            if os.path.exists(scaler_path):
                self.scaler = np.load(scaler_path, allow_pickle=True).item()
                logger.info("Model and scaler loaded successfully")
                return True
            else:
                logger.warning("Scaler file not found")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False 
