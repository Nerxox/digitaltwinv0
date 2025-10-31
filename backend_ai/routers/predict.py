from uuid import uuid4
import os
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Response, Header, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use shared predictor implementation (no FastAPI dependency inside)
from backend_ai.services.lstm_predictor import LSTMPredictor
from backend_ai.services.model_registry import ModelRegistry
from backend_ai.services.best_config import get_best_config, best_config_dict
from backend_ai.common.api_models import (
    PredictRequest,
    PredictResponse,
    MachineForecast,
    ForecastPoint,
    ApiError,
)

router = APIRouter()

# Supported horizons - can be made configurable later
SUPPORTED_H = [1, 5, 15, 30]

def horizons_list(h):
    return h if isinstance(h, list) else [h]

def machine_list(m):
    return m if isinstance(m, list) else [m]

@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ApiError},
        404: {"model": ApiError},
        503: {"model": ApiError},
    },
    tags=["predictions"],
)
async def predict(request: PredictRequest, response: Response, x_request_id: str | None = Header(default=None)):
    """Predict energy/power usage for machine(s) at one or more horizons."""
    try:
        # Headers for observability/cache policy
        rid = x_request_id or str(uuid4()) if 'uuid4' in globals() else "generated"
        response.headers["X-Request-ID"] = rid
        response.headers["Cache-Control"] = "no-store"

        # Validate horizons
        bad_horizons = [h for h in horizons_list(request.horizon_min) if h not in SUPPORTED_H]
        if bad_horizons:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ApiError(
                    code="HORIZON_UNSUPPORTED",
                    error="One or more horizons are not supported.",
                    details={"unsupported": bad_horizons, "supported": SUPPORTED_H},
                ).model_dump()
            )

        # Validate machines (basic check - replace with real registry lookup)
        machines = machine_list(request.machine_id)
        unknown_machines = [m for m in machines if not m.startswith("M")]
        if unknown_machines:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=ApiError(
                    code="NOT_FOUND",
                    error="One or more machines were not found.",
                    details={"unknown": unknown_machines},
                ).model_dump()
            )

        now = datetime.now(timezone.utc)
        want_quantiles = request.want_quantiles

        results = []
        for mid in machines:
            # Get model configuration
            best = get_best_config()
            algo = (request.algo or best.algo or "lstm").lower()
            scope = request.scope

            # Initialize model registry/predictor
            registry = ModelRegistry()
            key = mid if scope == "per_machine" else "global"
            demo_mode = os.getenv("DEMO_MODE", "false").lower() in {"1","true","yes","on"}
            try:
                predictor = registry.get(algo, key)
                predictor.load_model()
            except Exception as e:
                logger.error(f"Failed to load model for key={key}, algo={algo}: {str(e)}")
                if not demo_mode:
                    return JSONResponse(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        content=ApiError(
                            code="MODEL_NOT_READY",
                            error="Model not ready or failed to load.",
                            details={"algo": algo, "machine_id": mid, "scope": scope},
                        ).model_dump()
                    )
                predictor = None  # will use synthetic fallback

            points = []
            for h in horizons_list(request.horizon_min):
                # Get base predictions from model
                try:
                    if predictor is not None:
                        base_predictions = predictor.predict(mid, getattr(predictor, "horizon", 60)) or []
                    else:
                        # Synthetic baseline: simple diurnal sine wave in W
                        import math
                        horizon_steps = 60
                        base_predictions = [
                            500.0 + 300.0 * math.sin(2*math.pi*(i/24.0)) for i in range(horizon_steps)
                        ]
                except Exception as e:
                    logger.error(f"Prediction failed for machine_id={mid}: {str(e)}")
                    base_predictions = [0.0] * 60

                # Scale to requested horizon (simplified version)
                if h < 60:
                    yhat = (base_predictions[0] if base_predictions else 0.0) * (h / 60.0)
                else:
                    full_hours = min(h // 60, len(base_predictions))
                    partial = (h % 60) / 60.0
                    acc = sum(base_predictions[:full_hours]) if full_hours > 0 else 0.0
                    if partial > 0 and full_hours < len(base_predictions):
                        acc += base_predictions[full_hours] * partial
                    yhat = acc if acc != 0.0 else (0.0 if base_predictions else 0.0)

                # Drift flag based on scaler bounds (if available)
                drift_flag = False
                if best.scaler_min is not None and yhat < best.scaler_min:
                    drift_flag = True
                if best.scaler_max is not None and yhat > best.scaler_max:
                    drift_flag = True
                if predictor is None and h >= 30:
                    # In demo mode, mark longer horizons as drift to highlight anomalies
                    drift_flag = True

                # Quantiles (stubbed - replace with real implementation)
                quantiles = None
                if want_quantiles:
                    # For now, create synthetic quantiles based on prediction
                    # In real implementation, this would come from model quantile outputs
                    quantiles = {f"q{q:g}".replace(".", "_"): yhat for q in request.quantile_levels or [0.1, 0.5, 0.9]}

                points.append(ForecastPoint(
                    t=now + timedelta(minutes=h),
                    horizon_min=h,
                    yhat=float(yhat),
                    drift_flag=drift_flag,
                    quantiles=quantiles
                ))

            results.append(MachineForecast(
                machine_id=mid,
                algo=algo,
                scope=scope,
                horizons_available=SUPPORTED_H,
                generated_at=now,
                unit="W",  # Changed from kW to W as per specification
                points=points
            ))

        # Return single result or list based on input
        result = results[0] if isinstance(request.machine_id, str) else results
        # Let FastAPI serialize datetime fields (ISO-8601) via response_model
        return PredictResponse(result=result)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ApiError(
                code="INTERNAL_ERROR",
                error="An unexpected error occurred.",
                details={"message": str(e)},
            ).model_dump()
        )


@router.get(
    "/predict",
    response_model=PredictResponse,
    tags=["predictions"],
)
async def predict_get(
    response: Response,
    machine_id: str = Query(..., description="Machine id or comma-separated list e.g. M001 or M001,M002"),
    horizon_min: str = Query(..., description="Horizon in minutes or comma-separated list e.g. 15 or 5,15,30"),
    algo: str = Query("lstm", pattern="^(lstm|gru|xgboost|rf)$"),
    scope: str = Query("per_machine", pattern="^(per_machine|global)$"),
    want_quantiles: bool = Query(False),
    quantile_levels: str | None = Query(None, description="Comma-separated quantiles e.g. 0.1,0.5,0.9"),
    x_request_id: str | None = Header(default=None),
):
    # Parse comma-separated values
    mids = [m.strip() for m in machine_id.split(",")] if "," in machine_id else machine_id
    hrs = [int(x) for x in horizon_min.split(",")] if "," in horizon_min else int(horizon_min)
    qs = None
    if quantile_levels:
        try:
            qs = [float(q) for q in quantile_levels.split(",")]
        except Exception:
            qs = None
    req = PredictRequest(
        machine_id=mids,
        horizon_min=hrs,
        algo=algo,
        scope=scope,
        want_quantiles=want_quantiles,
        quantile_levels=qs,
    )
    return await predict(req, response, x_request_id)

@router.get("/model_info", response_model=dict, tags=["predictions"])
async def model_info():
    """Return the currently loaded best model configuration with metadata."""
    info = best_config_dict()
    return {
        "algo": str(info.get("algo", "lstm")),
        "lookback": int(info.get("lookback", 24)),
        "horizon": int(info.get("horizon", 1)),
        "scaler_min": info.get("scaler_min"),
        "scaler_max": info.get("scaler_max"),
        "source_path": str(info.get("source_path", "")),
        "timestamp": info.get("timestamp"),
    }