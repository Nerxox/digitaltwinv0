from typing import List, Dict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query
import numpy as np

from backend_ai.common.api_models import (
    AnomaliesRequest,
    AnomaliesResponse,
    AnomalyItem,
    MaintenanceRequest,
    MaintenanceResponse,
    MaintenanceRecItem,
    CompareRequest,
    CompareResponse,
)
from backend_ai.services.model_registry import ModelRegistry
from backend_ai.routers.evaluate import _load_split, _fetch_series_with_ts
from backend_ai.services.ga_optimizer import GADigitalTwinOptimizer

router = APIRouter(tags=["insights"]) 


def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mape = float(np.mean(np.abs((y_pred - y_true) / (y_true + 1e-6))) * 100.0)
    smape = float(np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6)) * 100.0)
    # Peak RMSE at top-10% values
    thr = float(np.quantile(y_true, 0.9))
    idx = y_true >= thr
    pk = float(np.sqrt(np.mean((y_pred[idx] - y_true[idx]) ** 2))) if idx.sum() > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape, "peak_rmse": pk}


@router.post("/anomalies", response_model=AnomaliesResponse)
async def anomalies(req: AnomaliesRequest):
    machines = req.machine_id if isinstance(req.machine_id, list) else ([req.machine_id] if req.machine_id else ["M001"])  # default
    horizons: List[int] = req.horizons or [1, 5, 15, 30]
    algo = (req.algo or "lstm").lower()

    registry = ModelRegistry()
    items: List[AnomalyItem] = []
    now = datetime.now(timezone.utc)

    for mid in machines:
        try:
            predictor = registry.get(algo, mid if req.scope == "per_machine" else "global")
            predictor.load_model()
        except Exception:
            # If model isn't ready, return empty points for that machine
            items.append(AnomalyItem(machine_id=mid, severity="Unknown", drifts=[], points=[]))
            continue

        pts = []
        drift_h = []
        for h in horizons:
            try:
                base_predictions = predictor.predict(mid, getattr(predictor, "horizon", 60)) or []
            except Exception:
                base_predictions = [0.0] * 60

            # Simplified scaling to horizon
            if h < 60:
                yhat = (base_predictions[0] if base_predictions else 0.0) * (h / 60.0)
            else:
                full_hours = min(h // 60, len(base_predictions))
                partial = (h % 60) / 60.0
                acc = sum(base_predictions[:full_hours]) if full_hours > 0 else 0.0
                if partial > 0 and full_hours < len(base_predictions):
                    acc += base_predictions[full_hours] * partial
                yhat = acc

            drift_flag = False
            # If scaler bounds are available on predictor, try to use them if attributes exist
            try:
                best_min = getattr(predictor, "scaler", None).data_min_[0] if getattr(predictor, "scaler", None) is not None else None
                best_max = getattr(predictor, "scaler", None).data_max_[0] if getattr(predictor, "scaler", None) is not None else None
            except Exception:
                best_min = best_max = None
            if best_min is not None and yhat < best_min:
                drift_flag = True
            if best_max is not None and yhat > best_max:
                drift_flag = True

            if drift_flag:
                drift_h.append(h)
            pts.append({"t": now + timedelta(minutes=h), "horizon_min": h, "yhat": float(yhat), "drift_flag": drift_flag})

        severity = "High" if any(h >= 30 for h in drift_h) else ("Medium" if drift_h else "None")
        items.append(AnomalyItem(machine_id=mid, severity=severity, drifts=drift_h, points=[]))

    return AnomaliesResponse(items=items)


@router.post("/maintenance/recommendations", response_model=MaintenanceResponse)
async def maintenance_recommendations(req: MaintenanceRequest):
    machines = req.machine_id if isinstance(req.machine_id, list) else ([req.machine_id] if req.machine_id else ["M001"])  # default
    algo = (req.algo or "lstm").lower()

    registry = ModelRegistry()
    items: List[MaintenanceRecItem] = []

    for mid in machines:
        # Evaluate a 15-min baseline if possible
        try:
            # Load test split and truth window
            splits = _load_split(mid)
            test = splits.get("boundaries", {}).get("test", {})
            t_end_idx = test.get("end_index")
            t_count = test.get("count", 0)
            ts, series = _fetch_series_with_ts(mid)
            if not series or t_count == 0:
                raise RuntimeError("no data")
            predictor = registry.get(algo, mid)
            predictor.load_model()
            steps = min(getattr(predictor, "horizon", 1), 15, t_count)
            y_true = np.asarray(series[t_end_idx - steps + 1 : t_end_idx + 1], dtype=float)
            y_pred = np.asarray(predictor.predict(mid, steps), dtype=float)
            mets = _calc_metrics(y_true, y_pred)
            rmse = float(mets["rmse"])
        except Exception:
            rmse = None

        # Drift check via anomalies endpoint logic (quick heuristic)
        drift = False
        try:
            predictor = registry.get(algo, mid)
            predictor.load_model()
            pts = predictor.predict(mid, getattr(predictor, "horizon", 60))
            drift = True if (pts and np.mean(pts) <= 0) else False
        except Exception:
            drift = False

        if drift and rmse and rmse >= 2.0:
            actions = [
                "Inspect recent process changes and loads",
                "Check sensor calibration and wiring",
                "Schedule maintenance and initiate model retraining",
            ]
            level = "High"
        elif drift and rmse and rmse >= 1.5:
            actions = [
                "Check sensors and environment (temperature, vibration)",
                "Review last shifts for abnormal operations",
                "Consider model refresh if persists",
            ]
            level = "Medium"
        elif drift:
            actions = [
                "Monitor machine for the next hour",
                "Log context (operator, batch, material)",
            ]
            level = "Low"
        else:
            actions = ["No immediate action required", "Continue routine checks"]
            level = "Normal"

        # --- Optimization Logic ---
        optimizer = GADigitalTwinOptimizer()
        optimal_cost, optimal_schedule = optimizer.get_optimal_schedule()
        
        # Add optimization recommendation to actions
        actions.append(f"Optimal 24h schedule cost: ${optimal_cost:.2f}")
        actions.append("Optimal Schedule Details (see response body for full table)")
        
        items.append(MaintenanceRecItem(
            machine_id=mid, 
            level=level, 
            rmse=rmse, 
            actions=actions,
            optimal_schedule=optimal_schedule, # Add the full schedule to the response model
            optimal_cost=optimal_cost
        ))

    return MaintenanceResponse(items=items)


@router.post("/compare", response_model=CompareResponse)
async def compare_models(req: CompareRequest):
    algo_list = [a.lower() for a in req.algos] if req.algos else ["lstm"]
    horizons = req.horizons or [5, 15, 30]
    mid = req.machine_id

    # Load truth window once
    splits = _load_split(mid)
    test = splits.get("boundaries", {}).get("test", {})
    t_end_idx = test.get("end_index")
    t_count = test.get("count", 0)
    ts, series = _fetch_series_with_ts(mid)
    if not series or t_count == 0:
        raise HTTPException(status_code=400, detail="No data for evaluation.")

    registry = ModelRegistry()
    out: Dict[str, Dict[str, Dict[str, float]]] = {}

    for algo in algo_list:
        try:
            predictor = registry.get(algo, mid)
            predictor.load_model()
        except Exception:
            continue

        algo_mets: Dict[str, Dict[str, float]] = {}
        for h in horizons:
            steps = int(max(1, min(getattr(predictor, "horizon", 1), h, t_count)))
            y_true = np.asarray(series[t_end_idx - steps + 1 : t_end_idx + 1], dtype=float)
            y_pred = np.asarray(predictor.predict(mid, steps), dtype=float)
            algo_mets[str(h)] = _calc_metrics(y_true, y_pred)
        out[algo] = algo_mets

    return CompareResponse(series=out)


# GET aliases for Swagger/Postman testing

@router.get("/anomalies", response_model=AnomaliesResponse)
async def anomalies_get(
    machine_id: str | None = Query(None, description="Comma-separated machine ids, e.g. M001,M002"),
    horizons: str | None = Query(None, description="Comma-separated horizons in minutes, e.g. 5,15,30"),
    algo: str = Query("lstm"),
    scope: str = Query("per_machine"),
):
    mids = [m.strip() for m in machine_id.split(",")] if machine_id else None
    hrs = [int(h) for h in horizons.split(",")] if horizons else None
    req = AnomaliesRequest(machine_id=mids if mids else None, horizons=hrs, algo=algo, scope=scope)
    return await anomalies(req)


@router.get("/maintenance/recommendations", response_model=MaintenanceResponse)
async def maintenance_recommendations_get(
    machine_id: str | None = Query(None, description="Comma-separated machine ids"),
    algo: str = Query("lstm"),
):
    mids = [m.strip() for m in machine_id.split(",")] if machine_id else None
    req = MaintenanceRequest(machine_id=mids if mids else None, algo=algo)
    return await maintenance_recommendations(req)


@router.get("/compare", response_model=CompareResponse)
async def compare_models_get(
    machine_id: str = Query(...),
    algos: str = Query("lstm", description="Comma-separated algos, e.g. lstm,gru,xgboost,rf"),
    horizons: str | None = Query(None, description="Comma-separated horizons in minutes, e.g. 5,15,30"),
    lookback: int = Query(24),
):
    algo_list = [a.strip() for a in algos.split(",") if a.strip()]
    hrs = [int(h) for h in horizons.split(",")] if horizons else None
    req = CompareRequest(machine_id=machine_id, algos=algo_list, horizons=hrs, lookback=lookback)
    return await compare_models(req)
