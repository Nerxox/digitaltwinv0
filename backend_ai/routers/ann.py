from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import os

# Optional runtimes
_ORT = None
_ORT_SESSION = None
_TORCH = None
_TORCH_MODEL = None

MODEL_PATH = os.getenv("ANN_MODEL_PATH")
MODEL_TYPE = (os.getenv("ANN_MODEL_TYPE") or "onnx").lower()  # onnx|torch

def _lazy_load_model():
    global _ORT, _ORT_SESSION, _TORCH, _TORCH_MODEL
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        return False
    try:
        if MODEL_TYPE == "onnx":
            import onnxruntime as ort
            _ORT = ort
            if _ORT_SESSION is None:
                _ORT_SESSION = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"]) 
            return True
        elif MODEL_TYPE == "torch":
            import torch
            _TORCH = torch
            if _TORCH_MODEL is None:
                _TORCH_MODEL = torch.jit.load(MODEL_PATH, map_location="cpu") if MODEL_PATH.endswith(".pt") else torch.load(MODEL_PATH, map_location="cpu")
                _TORCH_MODEL.eval()
            return True
    except Exception:
        return False
    return False


router = APIRouter(tags=["Anomaly Detection (ANN)"])


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
    thr = 3.0
    conf = float(min(0.99, 0.5 + min(z, 4.0) / 8.0))
    return z, thr, conf


@router.post("/ann/anomaly_score", response_model=InferenceResult)
def anomaly_score(req: InferenceRequest):
    """Run ANN model if configured; otherwise fallback to z-score.
    Expects a univariate recent window; the exported model should accept shape [1, T] or [1, T, 1].
    """
    used_model = False
    z, thr, conf = 0.0, 3.0, 0.5
    if _lazy_load_model():
        ys = np.array([p.y for p in req.series], dtype=np.float32)
        if ys.size:
            try:
                if _ORT_SESSION is not None:
                    x = ys.reshape(1, -1, 1)
                    feeds = {_ORT_SESSION.get_inputs()[0].name: x}
                    out = _ORT_SESSION.run(None, feeds)[0]
                    z = float(np.asarray(out).ravel()[-1])
                    thr = float(os.getenv("ANN_THRESHOLD", "3.0"))
                    conf = float(min(0.99, 0.5 + abs(z)/8.0))
                    used_model = True
                elif _TORCH_MODEL is not None:
                    x = _TORCH.tensor(ys).view(1, -1, 1).float()
                    with _TORCH.no_grad():
                        out = _TORCH_MODEL(x)
                        z = float(out.view(-1)[-1].item())
                    thr = float(os.getenv("ANN_THRESHOLD", "3.0"))
                    conf = float(min(0.99, 0.5 + abs(z)/8.0))
                    used_model = True
            except Exception:
                used_model = False
    if not used_model:
        z, thr, conf = _simple_zscore(req.series)
    return InferenceResult(
        machine_id=req.machine_id,
        score=z,
        drift_flag=bool(z >= thr),
        threshold=thr,
        confidence=conf,
    )


