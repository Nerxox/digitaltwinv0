"""Resample & dataset exporter.

Loads raw readings from DB for a machine, resamples to 1-minute frequency with forward
fill (limit=5), and exports:
  - datasets/train.csv
  - datasets/test.csv
  - datasets/schema_report.json (KPIs and schema)

Train/test split: chronological 80/20 by time (no leakage).

Usage:
    python -m backend_ai.data.splits --machine MACHINE_001
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from backend_ai.common.db import SessionLocal, init_db
from backend_ai.models import EnergyReading


@dataclass
class KPI:
    total_kwh: float
    peak_kw: float
    rows: int
    columns: List[str]


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    # Serialize with timezone if present
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)


def create_splits(machine_id: str) -> Dict:
    """Resample to 1T, ffill(limit=5), and export train/test CSVs and schema KPI JSON."""
    init_db()
    s = SessionLocal()
    try:
        rows = (
            s.query(EnergyReading.ts, EnergyReading.power_kw)
            .filter(EnergyReading.machine_id == machine_id)
            .order_by(EnergyReading.ts.asc())
            .all()
        )
    finally:
        s.close()

    if not rows:
        raise ValueError(f"No readings for machine_id={machine_id}")

    df = pd.DataFrame(rows, columns=["ts", "power_kw"])
    df = df.set_index(pd.to_datetime(df["ts"], utc=True)).drop(columns=["ts"])  # ensure UTC index
    # Resample to minutely with mean, then ffill limit=5
    df = df.resample("1T").mean()
    df["power_kw"] = df["power_kw"].ffill(limit=5)
    df = df.dropna()

    if len(df) < 10:
        raise ValueError("Insufficient rows after resample/ffill")

    # 80/20 split by time
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # KPIs
    minutes_energy_kwh = float(df["power_kw"].sum() / 60.0)
    peak_kw = float(df["power_kw"].max())
    schema = {
        "machine_id": machine_id,
        "kpis": {
            "total_kwh": minutes_energy_kwh,
            "peak_kw": peak_kw,
            "rows": int(len(df)),
            "columns": list(df.columns),
        },
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_ratio": float(len(train_df) / len(df)),
        "generated_at": _iso(datetime.utcnow()),
    }

    out_dir = Path("datasets")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    schema_path = out_dir / "schema_report.json"

    # Persist
    train_df.to_csv(train_path, index_label="ts")
    test_df.to_csv(test_path, index_label="ts")
    import json
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "schema_path": str(schema_path),
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Resample to 1T and export train/test CSVs and schema KPIs")
    parser.add_argument("--machine", required=True, help="Machine ID to export")
    args = parser.parse_args(argv)

    out = create_splits(args.machine)
    print(f"Wrote: {out['train_path']}, {out['test_path']}, {out['schema_path']}")


if __name__ == "__main__":
    main()
