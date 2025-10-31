import os
import logging
from typing import List

import joblib
import numpy as np

logger = logging.getLogger(__name__)

class TreePredictor:
    """
    Wrapper for tree-based regressors (RandomForest, XGBoost, etc.) trained to
    forecast the next value given a fixed-size lookback window.

    Expected artifacts:
      - model at `<model_base_path>.joblib`
      - scaler at `scaler_path`
    """
    def __init__(self, model_base_path='models/tree_energy', scaler_path='models/scaler.pkl', lookback: int = 24, horizon: int = 6):
        self.model_path = f"{model_base_path}.joblib"
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.lookback = lookback
        self.horizon = horizon

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

    def _recent_window(self, machine_id: str, lookback: int) -> np.ndarray:
        try:
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
                    return np.random.rand(lookback)
                data = np.array([float(r[0]) for r in rows], dtype=np.float32)[::-1]
                return data
            finally:
                db.close()
        except Exception:
            return np.random.rand(lookback)

    def predict(self, machine_id: str, steps: int) -> List[float]:
        if self.model is None or self.scaler is None:
            self.load_model()
        steps = min(steps, self.horizon)
        window = self._recent_window(machine_id, self.lookback).astype(np.float32)
        preds = []
        for _ in range(steps):
            x = window.reshape(1, -1)
            x_scaled = self.scaler.transform(x)
            yhat = float(self.model.predict(x_scaled)[0])
            preds.append(yhat)
            # roll window
            window = np.roll(window, -1)
            window[-1] = yhat
        return preds
