from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

try:
    from xgboost import XGBRegressor
except Exception as e:  # pragma: no cover
    XGBRegressor = None  # type: ignore


class XGBPredictor:
    """
    Simple XGBoost regressor for one-step ahead forecasting.
    Expects tabular features; use flattened lookback window as features.
    """

    def __init__(self, n_estimators: int = 300, max_depth: int = 4, learning_rate: float = 0.05, subsample: float = 0.8):
        if XGBRegressor is None:
            raise ImportError("xgboost is required for XGBPredictor. Please install 'xgboost'.")
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Tuple[float, int]:
        start = time.time()
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        end = time.time()
        return end - start, self.model.get_params().get("n_estimators", 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return preds.reshape(-1)
