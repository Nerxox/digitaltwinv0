from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class BestModelConfig:
    algo: str = "lstm"
    lookback: int = 24
    horizon: int = 1
    scaler_min: Optional[float] = None
    scaler_max: Optional[float] = None
    timestamp: Optional[str] = None
    source_path: Optional[str] = None


_BEST: Optional[BestModelConfig] = None


def load_winning_config(path: Path | None = None) -> Optional[BestModelConfig]:
    global _BEST
    if path is None:
        path = Path("bench") / "results" / "winning_config.json"
    if not path.exists():
        _BEST = BestModelConfig(source_path=str(path), timestamp=None)
        return _BEST
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        top = None
        if isinstance(data, dict):
            tm = data.get("top_models") or []
            if tm:
                top = tm[0]
        algo = (top or {}).get("algo", data.get("algo", "lstm"))
        lookback = int(data.get("lookback", 24))
        horizon = int(data.get("horizon", 1))
        scaler_min = data.get("scaler_min")
        scaler_max = data.get("scaler_max")
        # timestamp from file mtime
        ts = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        _BEST = BestModelConfig(
            algo=str(algo),
            lookback=lookback,
            horizon=horizon,
            scaler_min=scaler_min,
            scaler_max=scaler_max,
            timestamp=ts,
            source_path=str(path),
        )
    except Exception:
        _BEST = BestModelConfig(source_path=str(path), timestamp=None)
    return _BEST


def get_best_config() -> BestModelConfig:
    global _BEST
    if _BEST is None:
        load_winning_config()
    # _BEST will be set now
    return _BEST  # type: ignore


def best_config_dict() -> Dict[str, Any]:
    return asdict(get_best_config())
