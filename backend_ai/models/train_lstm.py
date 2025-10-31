from __future__ import annotations

import argparse
import json
import os
import tensorflow as tf
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Tuple, List

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


ART_MODELS = Path("artifacts") / "models"
ART_METRICS = Path("artifacts") / "metrics"
ART_FIGS = Path("artifacts") / "figures"
ART_PREDS = Path("artifacts") / "predictions"


def _now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = Path("datasets")
    train = pd.read_csv(d / "train.csv", parse_dates=["ts"], index_col="ts")
    test = pd.read_csv(d / "test.csv", parse_dates=["ts"], index_col="ts")
    # Ensure numeric
    train["power_kw"] = pd.to_numeric(train["power_kw"], errors="coerce")
    test["power_kw"] = pd.to_numeric(test["power_kw"], errors="coerce")
    train = train.dropna()
    test = test.dropna()
    return train, test


def build_features(df: pd.DataFrame, extra_lags: list[int] | None = None) -> pd.DataFrame:
    df = df.copy()
    
    # Basic lags
    lags = [1, 5, 15, 30, 60, 1440]  # 1min, 5min, 15min, 30min, 1h, 1d
    if extra_lags:
        lags += [int(x) for x in extra_lags if x not in lags]
    
    # Add lagged features
    for l in sorted(set(lags)):
        if l < len(df):  # Ensure lag is within bounds
            df[f"lag_{l}"] = df["power_kw"].shift(l)
    
    # Rolling statistics
    windows = [15, 60, 1440]  # 15min, 1h, 1d
    for w in windows:
        df[f"roll{w}_mean"] = df["power_kw"].rolling(w, min_periods=1).mean()
        df[f"roll{w}_std"] = df["power_kw"].rolling(w, min_periods=1).std().fillna(0.0)
        df[f"roll{w}_min"] = df["power_kw"].rolling(w, min_periods=1).min()
        df[f"roll{w}_max"] = df["power_kw"].rolling(w, min_periods=1).max()
    
    # Time-based features
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)
    df["month"] = df.index.month
    
    # Cyclical encoding for time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow']/7.0)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow']/7.0)
    
    # Remove any remaining NaN values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df


def make_supervised(df: pd.DataFrame, horizon_min: int) -> Tuple[pd.DataFrame, pd.Series]:
    y = df["power_kw"].shift(-horizon_min)  # minutes ahead since data is 1T
    X = df.drop(columns=["power_kw"])  # features only
    out = pd.concat([X, y.rename("target")], axis=1).dropna()
    return out.drop(columns=["target"]), out["target"]


def to_lstm_input(X: np.ndarray) -> np.ndarray:
    # Shape (n_samples, timesteps=1, n_features)
    return X.reshape((X.shape[0], 1, X.shape[1]))


def build_model(n_features: int) -> Sequential:
    m = Sequential([
        LSTM(64, return_sequences=True, input_shape=(1, n_features)),
        LSTM(32, return_sequences=False),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m


def plot_loss(history, horizon: int, tag: str) -> Path:
    ART_FIGS.mkdir(parents=True, exist_ok=True)
    p = ART_FIGS / f"loss_curve_h{horizon}_{tag}.pdf"
    plt.figure(figsize=(7,4))
    plt.plot(history.history.get("loss", []), label="train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Loss curve (h={horizon}m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p, format="pdf")
    plt.close()
    return p


def plot_error_timeseries(ts: pd.DatetimeIndex, abs_err: np.ndarray, horizon: int, tag: str) -> Path:
    ART_FIGS.mkdir(parents=True, exist_ok=True)
    p = ART_FIGS / f"error_timeseries_h{horizon}_{tag}.pdf"
    plt.figure(figsize=(10,3))
    plt.plot(ts, abs_err, lw=1, color="#d62728")
    plt.title(f"Absolute Error over Time (h={horizon}m)")
    plt.xlabel("Time")
    plt.ylabel("|Error| (kW)")
    plt.tight_layout()
    plt.savefig(p, format="pdf")
    plt.close()
    return p


def plot_pred_vs_actual(ts: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray, horizon: int, tag: str) -> Path:
    ART_FIGS.mkdir(parents=True, exist_ok=True)
    p = ART_FIGS / f"pred_vs_actual_h{horizon}_{tag}.pdf"
    plt.figure(figsize=(10,4))
    plt.plot(ts, y_true, label="Actual", lw=2)
    plt.plot(ts, y_pred, label="Predicted", lw=2, ls="--")
    plt.title(f"Prediction vs Actual (h={horizon}m)")
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p, format="pdf")
    plt.close()
    return p


def plot_daily_mean_power(df: pd.DataFrame, horizon: int, tag: str) -> Path:
    ART_FIGS.mkdir(parents=True, exist_ok=True)
    p = ART_FIGS / f"daily_mean_power_h{horizon}_{tag}.pdf"
    daily = df["power_kw"].resample("1D").mean()
    plt.figure(figsize=(10,4))
    plt.plot(daily.index, daily.values, lw=2)
    plt.title(f"Daily Mean Power (h={horizon}m)")
    plt.xlabel("Day")
    plt.ylabel("kW")
    plt.tight_layout()
    plt.savefig(p, format="pdf")
    plt.close()
    return p


def train_for_horizon(train_df: pd.DataFrame, test_df: pd.DataFrame, horizon: int, tag: str, *, peaks_variant: bool = False, peak_quantile: float = 0.95, weight_high: float = 3.0, extra_lags_cfg: list[int] | None = None) -> Dict:
    # Feature engineering
    if peaks_variant:
        extra_lags = extra_lags_cfg if extra_lags_cfg else [30, 60]
    else:
        extra_lags = extra_lags_cfg or []
    tr = build_features(train_df, extra_lags=extra_lags)
    te = build_features(test_df, extra_lags=extra_lags)

    Xtr_df, ytr_s = make_supervised(tr, horizon)
    Xte_df, yte_s = make_supervised(te, horizon)

    # Align
    if len(Xtr_df) == 0 or len(Xte_df) == 0:
        raise ValueError(f"No samples after feature construction for horizon={horizon}")

    # Scale
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_df.values.astype(np.float32))
    Xte = scaler.transform(Xte_df.values.astype(np.float32))

    # LSTM input
    Xtr_lstm = to_lstm_input(Xtr)
    Xte_lstm = to_lstm_input(Xte)

    # Model
    model = build_model(n_features=Xtr.shape[1])
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning rate scheduler
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )
    
    callbacks = [early_stop, reduce_lr]
    
    # Sample weighting for peak values
    sample_weight = None
    if peaks_variant:
        # Compute weights emphasizing high targets in train set
        thr = float(np.quantile(ytr_s.values, peak_quantile))
        w = np.ones_like(ytr_s.values, dtype=np.float32)
        w[ytr_s.values >= thr] = weight_high
        sample_weight = w
    
    # Training with validation split
    hist = model.fit(
        Xtr_lstm,
        ytr_s.values.astype(np.float32),
        validation_split=0.2,  # Increased validation split
        epochs=200,  # Increased max epochs
        batch_size=128,  # Larger batch size
        verbose=1,  # Show progress
        callbacks=callbacks,
        sample_weight=sample_weight,
        shuffle=True  # Shuffle training data
    )

    # Predict
    y_pred = model.predict(Xte_lstm, verbose=0).reshape(-1)
    y_true = yte_s.values.astype(np.float32)

    # Metrics
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    # sMAPE
    denom = (np.abs(y_true) + np.abs(y_pred))
    smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / np.where(denom == 0, 1.0, denom)) * 100.0)
    # Peak-aware metrics on test
    peak_thr = float(np.quantile(y_true, 0.95))
    true_peaks = np.where(y_true >= peak_thr)[0]
    pred_peaks = np.where(y_pred >= peak_thr)[0]
    tol = 2  # ±2 minutes tolerance
    hits = 0
    used = set()
    for i in true_peaks:
        match = [j for j in pred_peaks if abs(int(j) - int(i)) <= tol and j not in used]
        if match:
            hits += 1
            used.add(match[0])
    peak_recall = float(hits / max(1, len(true_peaks)))
    peak_precision = float(hits / max(1, len(pred_peaks))) if len(pred_peaks) else 0.0

    # Save artifacts
    ART_MODELS.mkdir(parents=True, exist_ok=True)
    model_path = ART_MODELS / f"lstm_h{horizon}_{tag}.keras"
    scaler_path = ART_MODELS / f"scaler_h{horizon}_{tag}.pkl"
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    # Figures
    loss_pdf = plot_loss(hist, horizon, tag)
    pred_pdf = plot_pred_vs_actual(Xte_df.index, y_true, y_pred, horizon, tag)
    daily_pdf = plot_daily_mean_power(pd.concat([train_df, test_df]), horizon, tag)
    err_pdf = plot_error_timeseries(Xte_df.index, np.abs(y_true - y_pred), horizon, tag)

    # Predictions CSV
    ART_PREDS.mkdir(parents=True, exist_ok=True)
    preds_path = ART_PREDS / f"predictions_h{horizon}_{tag}.csv"
    pd.DataFrame({
        "ts": Xte_df.index,
        "actual": y_true,
        "pred": y_pred,
        "error": (y_true - y_pred),
        "abs_error": np.abs(y_true - y_pred),
        "pct_error": np.where(y_true == 0, np.nan, (y_true - y_pred) / np.abs(y_true) * 100.0),
    }).to_csv(preds_path, index=False)

    return {
        "horizon": horizon,
        "mape": mape,
        "rmse": rmse,
        "mae": mae,
        "smape": smape,
        "peak_recall": peak_recall,
        "peak_precision": peak_precision,
        "peak_threshold": peak_thr,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "predictions_csv": str(preds_path),
        "figures": {
            "loss": str(loss_pdf),
            "pred_vs_actual": str(pred_pdf),
            "daily_mean_power": str(daily_pdf),
            "error_timeseries": str(err_pdf),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM models for multiple horizons using engineered features")
    parser.add_argument("--horizons", default="1,5,15,30", help="Comma-separated minute horizons")
    parser.add_argument("--peaks-variant", action="store_true", help="Enable peak-aware variant: weighted loss + extra lags")
    parser.add_argument("--peak-quantile", type=float, default=0.95, help="Quantile for peak weighting (0-1)")
    parser.add_argument("--weight-high", type=float, default=3.0, help="Sample weight for peak targets")
    parser.add_argument("--extra-lags", default="", help="Comma-separated extra lags to add (e.g., 30,60)")
    args = parser.parse_args()

    tag = _now_tag()
    train_df, test_df = load_datasets()

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    results: List[Dict] = []

    extra_lags_cfg = [int(x) for x in args.extra_lags.split(',') if x.strip().isdigit()] if args.extra_lags else None
    for h in horizons:
        r = train_for_horizon(
            train_df,
            test_df,
            h,
            tag,
            peaks_variant=args.peaks_variant,
            peak_quantile=args.peak_quantile,
            weight_high=args.weight_high,
            extra_lags_cfg=extra_lags_cfg,
        )
        print(f"H{h}m: MAPE={r['mape']:.2f}%, RMSE={r['rmse']:.3f} kW")
        results.append(r)

    # Metrics JSON
    ART_METRICS.mkdir(parents=True, exist_ok=True)
    metrics_path = ART_METRICS / f"metrics_{tag}.json"
    payload = {
        "timestamp": tag,
        "results": results,
        "summary": {
            "best_by_mape": sorted(results, key=lambda x: x["mape"])[0],
            "best_by_rmse": sorted(results, key=lambda x: x["rmse"])[0],
            "best_by_peak_recall": sorted(results, key=lambda x: x["peak_recall"], reverse=True)[0],
            "best_by_peak_precision": sorted(results, key=lambda x: x["peak_precision"], reverse=True)[0],
            "peaks_variant": args.peaks_variant,
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
