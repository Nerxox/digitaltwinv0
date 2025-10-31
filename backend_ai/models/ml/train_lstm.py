import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import joblib
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import database configuration
from backend_ai.common.db import SessionLocal
from backend_ai.models.energy_reading import EnergyReading

class LSTMForecaster:
    def __init__(self, 
                 lookback=3600,  # 60 minutes at 1Hz
                 horizon=600,    # 10 minutes at 1Hz
                 test_size=0.2,
                 batch_size=32,
                 epochs=10,
                 model_path='models/lstm_energy.h5',
                 scaler_path='models/scaler.pkl'):
        self.lookback = lookback
        self.horizon = horizon
        self.test_size = test_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_model()
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    def _build_model(self):
        """Build and compile the LSTM model."""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(self.horizon)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def load_data(self, machine_id='M1', hours=24):
        """Load data from SQLite database for a specific machine."""
        import sqlite3
        from pathlib import Path
        
        # Get the path to the SQLite database
        db_path = Path(__file__).parent.parent.parent / 'sql_app.db'
        
        # Connect to SQLite
        conn = sqlite3.connect(str(db_path))
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        query = """
        SELECT ts, power_kw 
        FROM energy_readings 
        WHERE machine_id = %s 
        AND ts BETWEEN %s AND %s
        ORDER BY ts
        """
        
        with conn:
            df = pd.read_sql_query(
                """
                SELECT ts, power_kw 
                FROM energy_readings 
                WHERE machine_id = ?
                AND ts BETWEEN ? AND ?
                ORDER BY ts
                """,
                conn,
                params=(machine_id, start_time.isoformat(), end_time.isoformat())
            )
        
        if df.empty:
            raise ValueError(f"No data found for machine {machine_id}")
            
        df['ts'] = pd.to_datetime(df['ts'])
        df.set_index('ts', inplace=True)
        return df['power_kw'].values.reshape(-1, 1)

    def create_sequences(self, data):
        """Create input sequences and targets for LSTM.
        
        Args:
            data: 2D numpy array with shape (n_samples, 1)
            
        Returns:
            X: 3D numpy array with shape (n_sequences, lookback, 1)
            y: 2D numpy array with shape (n_sequences, horizon)
        """
        X, y = [], []
        n_samples = len(data)
        
        # Ensure we have enough data to create at least one sequence
        if n_samples < self.lookback + self.horizon:
            return np.array(X), np.array(y)
            
        for i in range(n_samples - self.lookback - self.horizon + 1):
            # Get the sequence of lookback values
            seq_x = data[i:(i + self.lookback)]
            # Get the sequence of horizon values that come after
            seq_y = data[i + self.lookback:i + self.lookback + self.horizon]
            
            X.append(seq_x)
            y.append(seq_y.reshape(-1))  # Flatten y to 1D
            
        return np.array(X), np.array(y)

    def train_test_split(self, X, y):
        """Split data into training and testing sets."""
        split = int((1 - self.test_size) * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        return X_train, X_test, y_train, y_test

    def evaluate(self, y_true, y_pred):
        """Calculate and log evaluation metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        logger.info(f"MAE: {mae:.4f} kW")
        logger.info(f"MAPE: {mape:.2f}%")
        return mae, mape

    def train(self, machine_id='M1'):
        """Train the LSTM model."""
        logger.info(f"Training LSTM model for machine {machine_id}...")
        
        # Load and prepare data
        data = self.load_data(machine_id)
        logger.info(f"Loaded data shape: {data.shape}")
        
        if len(data) < self.lookback + self.horizon:
            raise ValueError(f"Not enough data points. Need at least {self.lookback + self.horizon} points, but got {len(data)}")
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        logger.info(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("No valid sequences created. Check your data and sequence parameters.")
        
        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split data
        X_train, X_test, y_train, y_test = self.train_test_split(X, y)
        logger.info(f"Train/Test split - X_train: {X_train.shape}, X_test: {X_test.shape}")
        
        # Train model
        logger.info(f"Training on {len(X_train)} samples...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.1,
            verbose=1
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        # Inverse transform predictions and actuals
        y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_inv = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mae, mape = self.evaluate(y_test_inv, y_pred_inv)
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save model and scaler
        model_path = self.model_path.replace('.h5', '.keras')  # Use .keras extension
        self.model.save(model_path, save_format='tf')
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Model saved to {model_path}")
        
        # Also save the model in H5 format for backward compatibility
        self.model.save(self.model_path, save_format='h5')
        logger.info(f"Model also saved in H5 format to {self.model_path}")
        
        return mae, mape

if __name__ == "__main__":
    # Adjust parameters to work with smaller dataset
    forecaster = LSTMForecaster(
        lookback=24,   # 24 hours of data points (assuming hourly data)
        horizon=6,     # Predict next 6 hours
        test_size=0.2,
        batch_size=8,  # Smaller batch size for smaller dataset
        epochs=50,     # More epochs since we have less data
        model_path='models/lstm_energy.h5',
        scaler_path='models/scaler.pkl'
    )
    mae, mape = forecaster.train('M1')
    print(f"Training complete. MAE: {mae:.2f} kW, MAPE: {mape:.2f}%")