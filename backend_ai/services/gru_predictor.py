from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


class GRUPredictor:
    """
    Simple GRU-based univariate forecaster.
    - lookback: number of past steps as input
    - horizon: one-step ahead by default for benchmarking (predict y[t])
    """

    def __init__(self, lookback: int = 24, hidden_units: int = 64, lr: float = 1e-3):
        self.lookback = lookback
        self.hidden_units = hidden_units
        self.lr = lr
        self.model = None

    def _build(self) -> None:
        m = Sequential([
            GRU(self.hidden_units, input_shape=(self.lookback, 1)),
            Dense(1),
        ])
        m.compile(optimizer=Adam(self.lr), loss="mse")
        self.model = m

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 32, patience: int = 8) -> Tuple[float, int]:
        if self.model is None:
            self._build()
        callbacks = []
        if X_val is not None and y_val is not None:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))
        start = time.time()
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        end = time.time()
        return end - start, epochs

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Model not trained"
        preds = self.model.predict(X, verbose=0)
        return preds.reshape(-1)
