from __future__ import annotations

from typing import Sequence, Dict
import numpy as np


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp))) if yt.size else float("nan")


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2))) if yt.size else float("nan")


def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    # Avoid division by zero: mask out zeros
    mask = yt != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)


def smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = (np.abs(yt) + np.abs(yp))
    mask = denom != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(2.0 * np.abs(yp[mask] - yt[mask]) / denom[mask]) * 100.0)


def all_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }
