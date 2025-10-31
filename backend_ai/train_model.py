"""Train a simple LSTM model for energy prediction."""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Model paths
MODEL_PATH = MODEL_DIR / "lstm_energy.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

def create_sequences(data, lookback, horizon):
    """Create sequences for LSTM."""
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback:i + lookback + horizon])
    return np.array(X), np.array(y)

def train_model():
    """Train an LSTM model on sample data."""
    try:
        # Generate sample data (sine wave with some noise)
        x = np.linspace(0, 20, 1000)
        y = 50 + 40 * np.sin(x) + np.random.normal(0, 5, len(x))
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(y.reshape(-1, 1))
        
        # Create sequences
        lookback = 24
        horizon = 6
        X, y = create_sequences(scaled_data, lookback, horizon)
        
        # Split into train/test
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Reshape for LSTM [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Build the model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(horizon)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Save the model and scaler
        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        
        logger.info(f"Model saved to {MODEL_PATH}")
        logger.info(f"Scaler saved to {SCALER_PATH}")
        
        return model, scaler
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

if __name__ == "__main__":
    print("Training a new LSTM model...")
    train_model()
    print("Training completed successfully!")
