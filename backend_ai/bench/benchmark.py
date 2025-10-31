from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from backend_ai.common.db import SessionLocal, init_db
from backend_ai.models import EnergyReading
from backend_ai.metrics import all_metrics
from backend_ai.services.gru_predictor import GRUPredictor
from backend_ai.services.xgb_predictor import XGBPredictor


@dataclass
class SplitIdx:
    train: Tuple[int, int]
    val: Tuple[int, int]
    test: Tuple[int, int]


def fetch_series(machine_id: str) -> Tuple[List[float], List]:
    """Fetch power series and timestamps for a machine, sorted ascending by ts."""
    init_db()
    s = SessionLocal()
    try:
        rows = (
            s.query(EnergyReading.ts, EnergyReading.power_kw)
            .filter(EnergyReading.machine_id == machine_id)
            .order_by(EnergyReading.ts.asc())
            .all()
        )
        ts = [r[0] for r in rows]
        y = [float(r[1]) for r in rows]
        return y, ts
    finally:
        s.close()


def list_all_machines() -> List[str]:
    """Return all distinct machine_ids in the database."""
    init_db()
    s = SessionLocal()
    try:
        rows = s.query(EnergyReading.machine_id).distinct().all()
        return [r[0] for r in rows]
    finally:
        s.close()


def compute_split_indices(ts: List, val_days: int, test_days: int) -> SplitIdx:
    if not ts:
        raise ValueError("No timestamps available for split")
    max_ts = ts[-1]
    n = len(ts)

    total_back = timedelta(days=val_days + test_days)
    test_back = timedelta(days=test_days)

    train_end = max_ts - total_back if (val_days + test_days) > 0 else max_ts
    val_start = train_end
    val_end = max_ts - test_back if test_days > 0 else max_ts
    test_start = val_end

    def idx_ge(target):
        for i, t in enumerate(ts):
            if t >= target:
                return i
        return None

    def idx_lt(target):
        for j in range(n - 1, -1, -1):
            if ts[j] < target:
                return j
        return None

    tr_s = 0
    tr_e = idx_lt(val_start)
    va_s = idx_ge(val_start)
    va_e = idx_lt(val_end)
    te_s = idx_ge(test_start)
    te_e = n - 1

    # Normalize Nones
    tr_e = tr_e if tr_e is not None else -1
    va_s = va_s if va_s is not None else n
    va_e = va_e if va_e is not None else -1
    te_s = te_s if te_s is not None else n

    return SplitIdx(train=(tr_s, tr_e), val=(va_s, va_e), test=(te_s, te_e))


def make_sequences(values: np.ndarray, lookback: int, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(values) - lookback - horizon + 1):
        X.append(values[i:i+lookback])
        y.append(values[i+lookback:i+lookback+horizon])
    X = np.array(X)
    y = np.array(y).reshape(-1, horizon)
    return X.reshape((-1, lookback, 1)), y.reshape((-1, horizon))


def subset_by_indices(X: np.ndarray, y: np.ndarray, idx: Tuple[int, int], lookback: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    s_idx, e_idx = idx
    # Map timestamp index range to sequence index range: sequence ends at i+lookback+horizon-1
    seq_start = max(0, s_idx - lookback)  # allow context before split start
    seq_end = min(len(X) - 1, e_idx - lookback - horizon + 1)
    if seq_end < seq_start:
        return X[0:0], y[0:0]
    return X[seq_start:seq_end+1], y[seq_start:seq_end+1]


def train_eval_lstm(X_train, y_train, X_val, y_val, X_test) -> Tuple[np.ndarray, float, float]:
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], 1)),
        Dense(1),
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse")
    cb = [EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)] if len(X_val) else []
    t0 = time.time()
    model.fit(X_train, y_train, validation_data=(X_val, y_val) if len(X_val) else None, epochs=100, batch_size=32, verbose=0, callbacks=cb)
    t1 = time.time()
    i0 = time.time()
    preds = model.predict(X_test, verbose=0).reshape(-1)
    i1 = time.time()
    infer_per_step = (i1 - i0) / max(1, len(X_test))
    return preds, (t1 - t0), infer_per_step


def train_eval_gru(X_train, y_train, X_val, y_val, X_test) -> Tuple[np.ndarray, float, float]:
    gru = GRUPredictor()
    train_time, _ = gru.fit(X_train, y_train, X_val, y_val)
    i0 = time.time()
    preds = gru.predict(X_test)
    i1 = time.time()
    infer_per_step = (i1 - i0) / max(1, len(X_test))
    return preds, train_time, infer_per_step


def train_eval_xgb(X_train, y_train, X_val, y_val, X_test) -> Tuple[np.ndarray, float, float]:
    Xtr = X_train.reshape((X_train.shape[0], -1))
    Xva = X_val.reshape((X_val.shape[0], -1)) if len(X_val) else None
    Xte = X_test.reshape((X_test.shape[0], -1))
    xgb = XGBPredictor()
    train_time, _ = xgb.fit(Xtr, y_train.reshape(-1), Xva, y_val.reshape(-1) if len(y_val) else None)
    i0 = time.time()
    preds = xgb.predict(Xte)
    i1 = time.time()
    infer_per_step = (i1 - i0) / max(1, len(X_test))
    return preds, train_time, infer_per_step


def benchmark(machine_id: str, val_days: int, test_days: int, lookback: int, out_dir: Path) -> Path:
    y_vals, ts_vals = fetch_series(machine_id)
    if len(y_vals) < lookback + 10:
        raise ValueError("Not enough data for benchmarking. Ingest more readings.")

    # Chronological splits
    splits = compute_split_indices(ts_vals, val_days, test_days)

    # Scale using train window only
    series = np.asarray(y_vals, dtype=np.float32).reshape(-1, 1)
    tr_s, tr_e = splits.train
    tr_e = max(tr_e, tr_s)
    scaler = MinMaxScaler()
    scaler.fit(series[tr_s:tr_e+1])
    series_scaled = scaler.transform(series)

    # Build sequences
    X_all, y_all = make_sequences(series_scaled, lookback, horizon=1)
    X_tr, y_tr = subset_by_indices(X_all, y_all, splits.train, lookback, 1)
    X_va, y_va = subset_by_indices(X_all, y_all, splits.val, lookback, 1)
    X_te, y_te = subset_by_indices(X_all, y_all, splits.test, lookback, 1)

    # Guard for empty splits
    if not len(X_tr) or not len(X_te):
        raise ValueError("Insufficient data in train/test splits after sequence construction.")

    # Train/Eval each algorithm
    records: List[Dict[str, float | str]] = []

    # LSTM
    lstm_preds_scaled, lstm_train_time, lstm_infer = train_eval_lstm(X_tr, y_tr, X_va, y_va, X_te)
    lstm_preds = scaler.inverse_transform(lstm_preds_scaled.reshape(-1, 1)).reshape(-1)

    # GRU
    gru_preds_scaled, gru_train_time, gru_infer = train_eval_gru(X_tr, y_tr, X_va, y_va, X_te)
    gru_preds = scaler.inverse_transform(gru_preds_scaled.reshape(-1, 1)).reshape(-1)

    # XGB
    xgb_preds_scaled, xgb_train_time, xgb_infer = train_eval_xgb(X_tr, y_tr, X_va, y_va, X_te)
    xgb_preds = scaler.inverse_transform(xgb_preds_scaled.reshape(-1, 1)).reshape(-1)

    # True labels (inverse)
    y_te_inv = scaler.inverse_transform(y_te).reshape(-1)

    def add_record(algo: str, preds: np.ndarray, train_time: float, infer_time: float):
        mets = all_metrics(y_te_inv, preds)
        rec = {
            "algo": algo,
            "MAE": mets["MAE"],
            "MAPE": mets["MAPE"],
            "RMSE": mets["RMSE"],
            "sMAPE": mets["sMAPE"],
            "train_time": float(train_time),
            "infer_time_per_step": float(infer_time),
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_va)),
            "n_test": int(len(X_te)),
            "lookback": int(lookback),
            "val_days": int(val_days),
            "test_days": int(test_days),
            "machine_id": machine_id,
        }
        records.append(rec)

    add_record("lstm", lstm_preds, lstm_train_time, lstm_infer)
    add_record("gru", gru_preds, gru_train_time, gru_infer)
    add_record("xgb", xgb_preds, xgb_train_time, xgb_infer)

    # Write CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"benchmark_{machine_id}_lb{lookback}_v{val_days}_t{test_days}.csv"
    fieldnames = [
        "algo", "MAE", "MAPE", "RMSE", "sMAPE", "train_time", "infer_time_per_step",
        "n_train", "n_val", "n_test", "lookback", "val_days", "test_days", "machine_id",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    return out_path


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark multiple algorithms on energy series")
    parser.add_argument("--machine", help="Single machine ID (deprecated in favor of --machines)")
    parser.add_argument("--machines", help="ALL or comma-separated machine IDs")
    parser.add_argument("--val-days", type=int, required=True)
    parser.add_argument("--test-days", type=int, required=True)
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--out-dir", default=str(Path("bench") / "results"))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve machine list
    machines: List[str] = []
    if args.machines:
        if args.machines.strip().upper() == "ALL":
            machines = list_all_machines()
        else:
            machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    elif args.machine:
        machines = [args.machine]
    else:
        raise SystemExit("Please provide --machines (ALL or list) or --machine")

    if not machines:
        raise SystemExit("No machines found to benchmark.")

    # Per-machine benchmarking
    per_machine_csvs: List[Path] = []
    metrics_by_machine_algo: Dict[str, Dict[str, Dict[str, float]]] = {}

    for mid in machines:
        try:
            csv_path = benchmark(mid, args.val_days, args.test_days, args.lookback, out_dir)
            per_machine_csvs.append(csv_path)
            # read back metrics
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    algo = row["algo"]
                    mets = {
                        "MAE": float(row["MAE"]),
                        "MAPE": float(row["MAPE"]),
                        "RMSE": float(row["RMSE"]),
                        "sMAPE": float(row["sMAPE"]),
                    }
                    metrics_by_machine_algo.setdefault(mid, {})[algo] = mets
        except Exception as e:
            # Still continue with others
            print(f"Warning: benchmark failed for {mid}: {e}")

    # Aggregate macro metrics across machines per algo
    algos = ["lstm", "gru", "xgb"]
    summary_rows: List[Dict[str, object]] = []
    import math

    def mean_std(values: List[float]) -> Tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        m = sum(values) / len(values)
        if len(values) > 1:
            var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
            sd = math.sqrt(var)
        else:
            sd = 0.0
        return m, sd

    # Collect by algo
    macro: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for algo in algos:
        coll = {k: [] for k in ["MAE", "MAPE", "RMSE", "sMAPE"]}
        for mid, per_algo in metrics_by_machine_algo.items():
            if algo in per_algo:
                for k in coll.keys():
                    v = per_algo[algo].get(k)
                    if v == v:  # not NaN
                        coll[k].append(v)
        macro[algo] = {k: mean_std(vals) for k, vals in coll.items()}

        # Prepare summary row
        row = {
            "algo": algo,
            "machines": len(coll["MAE"]),
        }
        for k, (m, sd) in macro[algo].items():
            row[f"{k}_mean"] = m
            row[f"{k}_std"] = sd
        summary_rows.append(row)

    # Write macro summary CSV
    summary_path = out_dir / f"summary_macro_lb{args.lookback}_v{args.val_days}_t{args.test_days}.csv"
    fieldnames = ["algo", "machines", "MAE_mean", "MAE_std", "MAPE_mean", "MAPE_std", "RMSE_mean", "RMSE_std", "sMAPE_mean", "sMAPE_std"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    # Determine top-1 and top-2 by MAE_mean (lower is better)
    ranked = sorted(summary_rows, key=lambda r: (r.get("MAE_mean", float("inf"))))
    top = ranked[:2]

    # Compute scaler ranges across all machines' train windows
    global_min = float("inf")
    global_max = float("-inf")
    for mid in machines:
        try:
            y_vals, ts_vals = fetch_series(mid)
            splits = compute_split_indices(ts_vals, args.val_days, args.test_days)
            tr_s, tr_e = splits.train
            tr_e = max(tr_e, tr_s)
            train_series = np.asarray(y_vals, dtype=float)[tr_s:tr_e+1]
            if train_series.size:
                global_min = min(global_min, float(np.min(train_series)))
                global_max = max(global_max, float(np.max(train_series)))
        except Exception:
            continue

    if global_min == float("inf"):
        global_min = float("nan")
    if global_max == float("-inf"):
        global_max = float("nan")

    # Save winning config JSON
    import json
    winning = {
        "lookback": int(args.lookback),
        "horizon": 1,
        "scaler_min": global_min,
        "scaler_max": global_max,
        "top_models": [
            {
                "rank": i + 1,
                "algo": r["algo"],
                "MAE_mean": r.get("MAE_mean"),
                "MAE_std": r.get("MAE_std"),
                "machines": r.get("machines", 0),
            }
            for i, r in enumerate(top)
        ],
    }
    win_path = out_dir / "winning_config.json"
    win_path.write_text(json.dumps(winning, indent=2), encoding="utf-8")

    # If only a single machine was requested, also print its CSV path
    if len(machines) == 1:
        print(f"Wrote benchmark results to {per_machine_csvs[0]}")
    print(f"Wrote macro summary to {summary_path}")
    print(f"Saved winning config to {win_path}")


if __name__ == "__main__":
    main()
