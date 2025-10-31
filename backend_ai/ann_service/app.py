from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import numpy as np


app = FastAPI(title="ANN Anomaly Service", version="0.1.0")


class SeriesPoint(BaseModel):
    t: Optional[float] = None
    y: float


class InferenceRequest(BaseModel):
    machine_id: str
    series: List[SeriesPoint]


class InferenceResult(BaseModel):
    machine_id: str
    score: float
    drift_flag: bool
    threshold: float
    confidence: float


def _simple_zscore(series: List[SeriesPoint]) -> tuple[float, float, float]:
    ys = np.array([p.y for p in series], dtype=float)
    if ys.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(ys))
    std = float(np.std(ys) + 1e-8)
    last = float(ys[-1])
    z = abs((last - mean) / std)
    # heuristic threshold
    thr = 3.0
    # pretend confidence grows with z but cap at 0.99
    conf = float(min(0.99, 0.5 + min(z, 4.0) / 8.0))
    return z, thr, conf


@app.post("/api/v1/ann/anomaly_score", response_model=InferenceResult)
def anomaly_score(req: InferenceRequest):
    # Placeholder: replace with loaded ANN model inference
    z, thr, conf = _simple_zscore(req.series)
    return InferenceResult(
        machine_id=req.machine_id,
        score=z,
        drift_flag=bool(z >= thr),
        threshold=thr,
        confidence=conf,
    )


