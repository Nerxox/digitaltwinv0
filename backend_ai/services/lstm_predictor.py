import os
import logging
from typing import List

import joblib
import numpy as np
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)

class LSTMPredictor:
    def __init__(self, model_base_path='models/lstm_energy', scaler_path='models/scaler.pkl'):
        self.model_path = f"{model_base_path}.keras"  # Prefer .keras format
        self.h5_model_path = f"{model_base_path}.h5"  # Fallback to .h5
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.lookback = 24  # Must match training
        self.horizon = 6    # Must match training

    def load_model(self):
        """Load the trained LSTM model and scaler."""
        if os.path.exists(self.model_path):
            logger.info(f"Loading model from {self.model_path}")
            self.model = load_model(self.model_path, compile=False)
        elif os.path.exists(self.h5_model_path):
            logger.info(f"Loading model from {self.h5_model_path}")
            self.model = load_model(self.h5_model_path, compile=False)
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path} or {self.h5_model_path}")

        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")

        self.scaler = joblib.load(self.scaler_path)
        logger.info("Model and scaler loaded successfully")

    def get_recent_data(self, machine_id: str, lookback: int) -> np.ndarray:
        """Get recent data using the project's SQLAlchemy session.
        Returns a (lookback, 1) numpy array ordered oldest->newest.
        Falls back to random data if unavailable.
        """
        try:
            # Lazy imports to avoid circular dependencies
            from backend_ai.common.db import SessionLocal
            from backend_ai.models.energy_reading import EnergyReading
            from sqlalchemy import desc

            db = SessionLocal()
            try:
                rows = (
                    db.query(EnergyReading.power_kw)
                    .filter(EnergyReading.machine_id == machine_id)
                    .order_by(desc(EnergyReading.ts))
                    .limit(lookback)
                    .all()
                )
                if not rows or len(rows) < lookback:
                    available = len(rows) if rows else 0
                    logger.warning(
                        f"Not enough data for machine {machine_id}. Need {lookback}, have {available}. Using fallback."
                    )
                    return np.random.rand(lookback, 1)

                # rows like [(val,), ...]; reverse to chronological order
                data = np.array([float(r[0]) for r in rows], dtype=np.float32)[::-1]
                return data.reshape(-1, 1)
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error retrieving recent data via SQLAlchemy: {e}", exc_info=True)
            return np.random.rand(lookback, 1)

    def predict(self, machine_id: str, steps: int) -> List[float]:
        """Generate predictions for the specified number of steps."""
        if not self.model or not self.scaler:
            self.load_model()

        if steps > self.horizon:
            steps = self.horizon

        # Fetch recent data
        recent_data = self.get_recent_data(machine_id, self.lookback)
        if recent_data is None:
            recent_data = np.random.rand(self.lookback, 1)

        # Ensure correct dtype
        recent_data = recent_data.astype(np.float32)

        # Scale and shape for LSTM
        scaled = self.scaler.transform(recent_data)
        X = scaled.reshape(1, self.lookback, 1)

        # Predict
        yhat = self.model.predict(X)

        # Slice requested steps and inverse-transform to original units
        pred_slice = yhat[0][:steps].reshape(-1, 1)
        pred_inv = self.scaler.inverse_transform(pred_slice).flatten()
        return pred_inv.tolist()
