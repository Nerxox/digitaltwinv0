import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import logging

# --- Configuration ---
LOOKBACK = 24  # Hours of historical data to look at
HORIZON = 6    # Hours to predict (currently unused in this simple model, but good practice)
EPOCHS = 50
BATCH_SIZE = 32
MODEL_NAME = "lstm_energy"
SCALER_NAME = "scaler.pkl"
DATA_PATH = "datasets/simulated_data.csv"
MODEL_DIR = "models"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sequences(data, lookback):
    """
    Create sequences of data for LSTM training.
    X is the lookback window, y is the next value.
    """
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback), 0])
        y.append(data[i + lookback, 0])
    return np.array(X), np.array(y)

def train_lstm_model():
    """
    Loads data, preprocesses, trains an LSTM model, and saves the model and scaler.
    """
    logger.info("Starting LSTM model training process...")

    # 1. Load Data
    try:
        df = pd.read_csv(DATA_PATH, index_col='timestamp', parse_dates=True)
        logger.info(f"Data loaded successfully from {DATA_PATH}. Shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Data file not found at {DATA_PATH}. Please ensure it exists.")
        return
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Use 'total_power_kw' as the target variable
    data = df[['total_power_kw']].values.astype('float32')

    # 2. Preprocessing (Scaling)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    logger.info("Data scaled using MinMaxScaler.")

    # 3. Create Sequences
    X, y = create_sequences(scaled_data, LOOKBACK)
    
    # Reshape X for LSTM input: [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    logger.info(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}")

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    logger.info(f"Data split. Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # 5. Build LSTM Model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1)) # Output layer predicts the next single value
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    logger.info("LSTM model compiled.")

    # 6. Train Model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    logger.info("Model training complete.")

    # 7. Save Model and Scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.keras")
    scaler_save_path = os.path.join(MODEL_DIR, SCALER_NAME)
    
    model.save(model_save_path)
    joblib.dump(scaler, scaler_save_path)
    
    logger.info(f"Model saved to: {model_save_path}")
    logger.info(f"Scaler saved to: {scaler_save_path}")
    logger.info("Training process finished successfully.")

if __name__ == "__main__":
    # Change directory to the project root to correctly resolve paths
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("..") # Move up to backend_ai
    os.chdir("..") # Move up to digitaltwinv0/digitaltwinv0
    
    # Check if data exists before training
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found at {DATA_PATH}. Please run the data simulator first.")
    else:
        train_lstm_model()
