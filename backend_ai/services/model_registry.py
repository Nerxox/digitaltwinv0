import os
from pathlib import Path
from typing import Dict, Tuple

from backend_ai.services.lstm_predictor import LSTMPredictor
from backend_ai.services.tree_predictor import TreePredictor


class ModelRegistry:
    """
    Lightweight registry to resolve per-machine and global model locations.
    - algo: supports 'lstm', 'gru' (treated like lstm loaders), 'xgboost', 'rf'
    - key: machine_id for per_machine, or 'global' for global scope
    
    Directory layout convention (customizable via env vars):
      models/
        lstm/
          global.keras, global.h5, scaler_global.pkl
          <machine_id>.keras, <machine_id>.h5, scaler_<machine_id>.pkl
    Fallbacks:
      - if per-machine model absent, can fall back to global if present
    """

    # simple process-wide cache {(algo,key): predictor}
    _CACHE: dict[tuple[str, str], object] = {}

    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path("models")
        self.algo_dirs: Dict[str, Path] = {
            "lstm": self.base_dir / "lstm",
            "gru": self.base_dir / "gru",
            "xgboost": self.base_dir / "xgboost",
            "rf": self.base_dir / "rf",
        }
        # Ensure directories exist (non-fatal if they don't)
        for p in self.algo_dirs.values():
            p.mkdir(parents=True, exist_ok=True)

    def _paths_for(self, algo: str, key: str) -> Tuple[Path, Path]:
        if algo not in self.algo_dirs:
            raise ValueError(f"Unsupported algo: {algo}")
        algo_dir = self.algo_dirs[algo]
        model_base = algo_dir / key
        model_base_path = str(model_base)
        # scaler beside with key suffix
        scaler_path = str(algo_dir / f"scaler_{key}.pkl")
        return Path(model_base_path), Path(scaler_path)

    def get(self, algo: str, key: str, allow_global_fallback: bool = True):
        """
        Return a configured predictor for the given algo and key.
        For 'lstm', returns LSTMPredictor with model_base_path and scaler_path set.
        
        Fallback: if per-machine files absent and allow_global_fallback is True,
        try 'global'.
        """
        # Fast path: return cached loaded predictor
        cache_key = (algo, key)
        pred = ModelRegistry._CACHE.get(cache_key)
        if pred is not None:
            return pred

        model_base_path, scaler_path = self._paths_for(algo, key)
        # Choose predictor implementation by algo
        if algo in {"lstm", "gru"}:
            predictor = LSTMPredictor(model_base_path=str(model_base_path), scaler_path=str(scaler_path))
        elif algo in {"xgboost", "rf"}:
            predictor = TreePredictor(model_base_path=str(model_base_path), scaler_path=str(scaler_path))
        else:
            raise ValueError(f"Unsupported algo: {algo}")

        # Resolve existence (without loading the model yet). We'll try to find either .keras or .h5
        keras_exists = Path(str(model_base_path) + ".keras").exists()
        h5_exists = Path(str(model_base_path) + ".h5").exists()
        joblib_exists = Path(str(model_base_path) + ".joblib").exists()
        scaler_exists = Path(scaler_path).exists()

        if not (keras_exists or h5_exists or joblib_exists) or not scaler_exists:
            # First fallback: global scoped under algo dir
            if allow_global_fallback and key != "global":
                g_base, g_scaler = self._paths_for(algo, "global")
                g_keras = Path(str(g_base) + ".keras").exists()
                g_h5 = Path(str(g_base) + ".h5").exists()
                g_joblib = Path(str(g_base) + ".joblib").exists()
                g_scaler_exists = Path(g_scaler).exists()
                if g_scaler_exists and (g_keras or g_h5 or g_joblib):
                    if algo in {"lstm", "gru"}:
                        predictor = LSTMPredictor(model_base_path=str(g_base), scaler_path=str(g_scaler))
                        ModelRegistry._CACHE[cache_key] = predictor
                        return predictor
                    else:
                        predictor = TreePredictor(model_base_path=str(g_base), scaler_path=str(g_scaler))
                        ModelRegistry._CACHE[cache_key] = predictor
                        return predictor

            # Second fallback: default legacy paths at models/lstm_energy.* with models/scaler.pkl
            default_base = Path("models") / ("lstm_energy" if algo in {"lstm","gru"} else "tree_energy")
            default_scaler = Path("models") / "scaler.pkl"
            if default_scaler.exists() and (
                Path(str(default_base) + ".keras").exists() or
                Path(str(default_base) + ".h5").exists() or
                Path(str(default_base) + ".joblib").exists()
            ):
                if algo in {"lstm","gru"}:
                    predictor = LSTMPredictor(model_base_path=str(default_base), scaler_path=str(default_scaler))
                    ModelRegistry._CACHE[cache_key] = predictor
                    return predictor
                else:
                    predictor = TreePredictor(model_base_path=str(default_base), scaler_path=str(default_scaler))
                    ModelRegistry._CACHE[cache_key] = predictor
                    return predictor
        # Cache and return default resolved predictor path; caller may call load_model once and reuse
        ModelRegistry._CACHE[cache_key] = predictor
        return predictor
