from typing import List, Dict, Union, Optional
from datetime import datetime
from pydantic import BaseModel, Field, conint, confloat
from typing_extensions import Literal

# Enums
Algo = Literal["lstm", "gru", "xgboost", "rf"]
Scope = Literal["global", "per_machine"]

# Enhanced models with validation and quantiles support

class PredictRequest(BaseModel):
    machine_id: Union[str, List[str]]
    horizon_min: Union[conint(ge=1), List[conint(ge=1)]]
    algo: str = Field(pattern="^(lstm|gru|xgboost|rf)$")
    scope: str = Field(pattern="^(per_machine|global)$")
    want_quantiles: bool = False
    quantile_levels: Optional[List[confloat(ge=0, le=1)]] = Field(default_factory=lambda: [0.1, 0.5, 0.9])

class ForecastPoint(BaseModel):
    t: datetime
    horizon_min: int
    yhat: float
    drift_flag: bool = False
    quantiles: Optional[Dict[str, float]] = None

class MachineForecast(BaseModel):
    machine_id: str
    algo: str
    scope: str
    horizons_available: List[int]
    generated_at: datetime
    unit: str = "W"
    points: List[ForecastPoint]

class PredictResponse(BaseModel):
    result: Union[MachineForecast, List[MachineForecast]]

class ApiError(BaseModel):
    code: str
    error: str
    details: Optional[Dict[str, object]] = None

# Keep existing models for backward compatibility during transition
class PredictMeta(BaseModel):
    model_version: str
    request_id: str
    generated_at_utc: str
    scaler_min: Optional[float] = None
    scaler_max: Optional[float] = None
    lookback: Optional[int] = None
    horizons_available: Optional[List[int]] = None
    drift_flag: bool = False

class ModelInfo(BaseModel):
    algo: Algo
    lookback: int
    horizon: int
    scaler_min: Optional[float] = None
    scaler_max: Optional[float] = None
    train_span: Optional[Dict[str, str]] = None
    git_sha: Optional[str] = None
    source_path: str
    timestamp: Optional[str] = None
    machines: Optional[List[str]] = None
    horizons_available: Optional[List[int]] = None

class EvalRequest(BaseModel):
    machine_id: str
    algo: Algo = "lstm"
    lookback: int = 24
    horizons: Optional[List[int]] = None
    start_utc: Optional[str] = None
    end_utc: Optional[str] = None

class EvalMetrics(BaseModel):
    """Metrics for a single model or baseline"""
    mae: float
    rmse: float
    mape: float
    smape: float
    r2: float
    nmae: float
    nrmse: float
    mase: float
    peak_rmse: Optional[float] = None

class EvalResponse(BaseModel):
    """Response model for evaluation endpoint with hierarchical metrics"""
    metrics: Dict[str, Dict[str, Union[Dict, float]]]
    n: int
    window_minutes: float
    figure_path: str
    csv_path: str

class Healthz(BaseModel):
    status: str
    version: str
    uptime_s: float
    now_utc: str
    hostname: str

# New API models for anomalies, maintenance, and comparisons

class AnomalyItem(BaseModel):
    machine_id: str
    severity: Literal["None", "Medium", "High", "Unknown"]
    drifts: List[int] = []
    points: Optional[List[ForecastPoint]] = None

class AnomaliesRequest(BaseModel):
    machine_id: Optional[Union[str, List[str]]] = None
    horizons: Optional[List[int]] = None
    algo: Algo = "lstm"
    scope: Scope = "per_machine"

class AnomaliesResponse(BaseModel):
    items: List[AnomalyItem]

class MaintenanceRecItem(BaseModel):
    machine_id: str
    level: Literal["Normal", "Low", "Medium", "High"]
    rmse: Optional[float] = None
    actions: List[str]

class MaintenanceRequest(BaseModel):
    machine_id: Optional[Union[str, List[str]]] = None
    algo: Algo = "lstm"

class MaintenanceResponse(BaseModel):
    items: List[MaintenanceRecItem]

class CompareRequest(BaseModel):
    machine_id: str
    algos: List[Algo] = ["lstm"]
    horizons: Optional[List[int]] = None
    lookback: int = 24

class CompareResponse(BaseModel):
    series: Dict[str, Dict[str, Dict[str, float]]]
