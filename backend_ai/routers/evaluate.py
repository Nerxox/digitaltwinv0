from typing import Dict, List
from pathlib import Path

import csv
import math
import numpy as np
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from backend_ai.metrics import all_metrics
from backend_ai.services.model_registry import ModelRegistry
from backend_ai.common.db import SessionLocal
from backend_ai.models.energy_reading import EnergyReading
from backend_ai.bench.plots import plot_real_vs_pred
from backend_ai.common.api_models import EvalRequest, EvalResponse

router = APIRouter()


def _load_split(machine_id: str) -> Dict:
    """Load split JSON if present; otherwise compute 80/20 split over raw series
    and persist to datasets/splits/<machine_id>.json.
    The indices correspond to the raw chronological series fetched by _fetch_series_with_ts().
    """
    split_path = Path("datasets") / "splits" / f"{machine_id}.json"
    if split_path.exists():
        import json
        return json.loads(split_path.read_text(encoding="utf-8"))

    # Fallback: compute boundaries from DB (raw series, not resampled)
    ts, series = _fetch_series_with_ts(machine_id)
    n = len(series)
    if n == 0:
        raise FileNotFoundError(
            f"Split file not found and no data available to compute it: {split_path}"
        )

    split_idx = int(n * 0.8)
    t_start_idx = split_idx
    t_end_idx = n - 1
    t_count = max(0, n - split_idx)

    payload = {
        "machine_id": machine_id,
        "boundaries": {
            "test": {
                "start_index": t_start_idx,
                "end_index": t_end_idx,
                "count": t_count,
            }
        },
        "generated_by": "backend_ai.routers.evaluate._load_split",
    }

    # Persist for future calls
    split_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    split_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _fetch_series_with_ts(machine_id: str):
    db = SessionLocal()
    try:
        rows = (
            db.query(EnergyReading.ts, EnergyReading.power_kw)
            .filter(EnergyReading.machine_id == machine_id)
            .order_by(EnergyReading.ts.asc())
            .all()
        )
        ts = [r[0] for r in rows]
        y = [float(r[1]) for r in rows]
        return ts, y
    finally:
        db.close()


def _peak_rmse(y_true: np.ndarray, y_pred: np.ndarray, p: float = 0.9) -> float:
    thr = float(np.quantile(y_true, p))
    idx = y_true >= thr
    if idx.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_pred[idx] - y_true[idx]) ** 2)))


@router.post("/evaluate", response_model=EvalResponse)
async def evaluate(req: EvalRequest):
    algo = (req.algo or "lstm").lower()
    supported = {"lstm", "gru", "xgboost", "rf"}
    if algo not in supported:
        raise HTTPException(status_code=400, detail=f"Unsupported algo '{algo}'. Supported: {sorted(supported)}")

    # Load precomputed test split (or auto-generate if missing)
    splits = _load_split(req.machine_id)
    test = splits.get("boundaries", {}).get("test", {})
    t_start_idx = test.get("start_index")
    t_end_idx = test.get("end_index")
    t_count = test.get("count", 0)
    if t_start_idx is None or t_end_idx is None or t_count == 0:
        raise HTTPException(status_code=400, detail="Test split not available or empty. Regenerate splits.")

    ts, series = _fetch_series_with_ts(req.machine_id)
    if not ts or not series:
        raise HTTPException(status_code=400, detail="No data found for machine.")

    # Determine horizons: support new list field if provided, else fallback to single lookahead via lookback/horizon
    horizons: List[int] = req.horizons if req.horizons else [getattr(req, "lookback", 24) and 1]

    # Model
    registry = ModelRegistry()
    # For 'gru' we reuse LSTMPredictor loader path convention
    predictor = registry.get(algo, req.machine_id)
    predictor.load_model()

    metrics_dict: Dict[str, Dict[str, float]] = {}
    max_steps_used = 0
    ts_true_for_plot: List = []
    y_true_for_plot: List[float] = []
    y_pred_for_plot: List[float] = []

    for h in horizons:
        steps = int(max(1, min(predictor.horizon, h, t_count)))
        y_true = np.asarray(series[t_end_idx - steps + 1 : t_end_idx + 1], dtype=float)
        ts_true = ts[t_end_idx - steps + 1 : t_end_idx + 1]
        y_pred = np.asarray(predictor.predict(req.machine_id, steps), dtype=float)

        # Compute metrics explicitly to avoid confusion
        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mape = float(np.mean(np.abs((y_pred - y_true) / (y_true + 1e-6))) * 100.0)
        smape = float(np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6)) * 100.0)
        pk = _peak_rmse(y_true, y_pred, 0.9)

        metrics_dict[str(h)] = {
            "mae": mae,
            "mape": mape,
            "smape": smape,
            "rmse": rmse,
            "peak_rmse": pk,
        }

        if steps > max_steps_used:
            max_steps_used = steps
            ts_true_for_plot = ts_true
            y_true_for_plot = y_true.tolist()
            y_pred_for_plot = y_pred.tolist()

    # Window minutes
    window_minutes = 0.0
    if ts_true_for_plot:
        window_minutes = (ts_true_for_plot[-1] - ts_true_for_plot[0]).total_seconds() / 60.0
        if window_minutes <= 0 and max_steps_used > 1:
            window_minutes = float(max_steps_used - 1)

    # Plot and save PDF (use the largest-horizon window for a single figure)
    fig_path = plot_real_vs_pred(
        machine_id=req.machine_id,
        algo=req.algo,
        timestamps=ts_true_for_plot,
        y_true=y_true_for_plot,
        y_pred=y_pred_for_plot,
    )

    # Write CSV snapshot
    out_dir = Path("artifacts") / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = out_dir / f"api_eval_{ts_now}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["machine_id", "algo", "lookback", "horizon", "mae", "mape", "smape", "rmse", "peak_rmse", "n", "window_minutes"])
        for h, mets in metrics_dict.items():
            writer.writerow([
                req.machine_id,
                req.algo,
                req.lookback,
                h,
                mets["mae"],
                mets["mape"],
                mets["smape"],
                mets["rmse"],
                mets["peak_rmse"],
                max_steps_used,
                float(window_minutes),
            ])

    return EvalResponse(
        metrics=metrics_dict,
        n=max_steps_used,
        window_minutes=float(window_minutes),
        figure_path=str(fig_path),
        csv_path=str(csv_path)  # Add this line
    )
