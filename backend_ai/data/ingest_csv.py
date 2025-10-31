from typing import List, Optional


def _load_env_if_present() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass


AUTO_TS = ["ts", "timestamp", "time", "datetime", "date", "event_time"]
AUTO_MACHINE = ["machine", "machine_id", "asset", "asset_id", "line", "equipment"]
AUTO_POWER = [
    "power",
    "power_kw",
    "kw",
    "kW",
    "p_kw",
    "consumption",
    "power_consumption",
    "energy_consumption",
    "energy_kw",
    "powerkW",
]
AUTO_STATE = ["state", "status", "machine_state", "op_state", "production_status"]
AUTO_THROUGHPUT = ["throughput", "qty", "quantity", "units", "output"]


def _detect(name_list: List[str], header: List[str]) -> Optional[str]:
    low = [h.lower() for h in header]
    for cand in name_list:
        if cand.lower() in low:
            return header[low.index(cand.lower())]
    return None

"""CSV ingestion utility to load energy readings into the database with auto-detection.

Usage:
    python -m backend_ai.data.ingest_csv --csv path/to/file.csv [--ts TS] [--machine MACH] [--power POW] [--state STATE] [--throughput TH]

Behavior:
- CLI args override auto-detection.
- Robust timestamp parsing; naive timestamps assumed UTC; 'Z' handled.
- Deduplicate on (timestamp, machine_id) against DB and within file.
"""
from datetime import datetime, timezone
import argparse
import csv
from pathlib import Path
ISO_FORMATS = [
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
]
def parse_ts(value: str) -> datetime:
    v = value.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        pass
    for fmt in ISO_FORMATS:
        try:
            dt = datetime.strptime(v, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            continue
    # Fallback: raise
    raise ValueError(f"Unrecognized timestamp format: {value}")


def ingest_csv(csv_path: Path, ts_col: Optional[str], machine_col: Optional[str], power_col: Optional[str], state_col: Optional[str] = None, throughput_col: Optional[str] = None, batch_size: int = 1000) -> int:
    # Lazy import DB modules here to avoid circular imports during tooling
    from backend_ai.common.db import SessionLocal, init_db  # type: ignore
    from backend_ai.models import EnergyReading  # type: ignore
    _load_env_if_present()
    init_db()  # ensure tables exist
    session = SessionLocal()
    total = 0
    batch: List[EnergyReading] = []
    skipped_duplicates = 0
    skipped_missing_power = 0
    skipped_bad_ts = 0
    scanned = 0
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header row")
            header = reader.fieldnames

            # Auto-detect if not provided
            ts_col = ts_col or _detect(AUTO_TS, header)
            machine_col = machine_col or _detect(AUTO_MACHINE, header)
            power_col = power_col or _detect(AUTO_POWER, header)
            state_col = state_col or _detect(AUTO_STATE, header)
            throughput_col = throughput_col or _detect(AUTO_THROUGHPUT, header)

            missing = [name for name, col in {"ts": ts_col, "machine": machine_col, "power": power_col}.items() if not col]
            if missing:
                raise ValueError(f"Unable to detect required columns: {missing}. Header={header}")

            # Preload existing keys to dedupe: gather all machine_ids in file and min/max ts
            # Build in-file cache too
            file_pairs: set[tuple[str, datetime]] = set()
            machines_in_file: set[str] = set()
            ts_min: Optional[datetime] = None
            ts_max: Optional[datetime] = None

            rows_cached: List[dict] = []
            for row in reader:
                rows_cached.append(row)
            # First pass compute meta
            for row in rows_cached:
                try:
                    tsv = parse_ts(str(row[ts_col]))  # type: ignore[arg-type]
                    mval = str(row[machine_col]).strip()  # type: ignore[index]
                    machines_in_file.add(mval)
                    ts_min = tsv if ts_min is None or tsv < ts_min else ts_min
                    ts_max = tsv if ts_max is None or tsv > ts_max else ts_max
                except Exception:
                    continue

            existing_pairs: set[tuple[str, datetime]] = set()
            if machines_in_file and ts_min and ts_max:
                for m in machines_in_file:
                    q = (
                        session.query(EnergyReading.machine_id, EnergyReading.ts)
                        .filter(EnergyReading.machine_id == m)
                        .filter(EnergyReading.ts >= ts_min)
                        .filter(EnergyReading.ts <= ts_max)
                        .all()
                    )
                    for mi, ti in q:
                        existing_pairs.add((str(mi), ti))

            # Second pass insert
            for row in rows_cached:
                scanned += 1
                try:
                    # Timestamp parse
                    ts_raw = row.get(ts_col) if ts_col else None  # type: ignore[arg-type]
                    if ts_raw is None or str(ts_raw).strip() == "":
                        skipped_bad_ts += 1
                        continue
                    ts = parse_ts(str(ts_raw))

                    machine_id = str(row[machine_col]).strip()  # type: ignore[index]

                    # Power parse: skip empty or non-numeric
                    p_raw = row.get(power_col)  # type: ignore[index]
                    if p_raw is None or str(p_raw).strip() == "":
                        skipped_missing_power += 1
                        continue
                    try:
                        power_kw = float(str(p_raw).replace(",", ""))
                    except Exception:
                        skipped_missing_power += 1
                        continue

                    status_val = str(row[state_col]).strip() if state_col and row.get(state_col) is not None else "RUN"  # type: ignore[index]

                    pair = (machine_id, ts)
                    if pair in file_pairs or pair in existing_pairs:
                        skipped_duplicates += 1
                        continue
                    file_pairs.add(pair)

                    item = EnergyReading(
                        machine_id=machine_id,
                        ts=ts,
                        power_kw=power_kw,
                        status=status_val or "RUN",
                    )
                    batch.append(item)
                    if len(batch) >= batch_size:
                        session.add_all(batch)
                        session.commit()
                        total += len(batch)
                        batch.clear()
                except Exception:
                    session.rollback()
                    raise
            if batch:
                session.add_all(batch)
                session.commit()
                total += len(batch)
    finally:
        session.close()
    print(
        f"Scanned: {scanned} | Inserted: {total} | Duplicates: {skipped_duplicates} | "
        f"Missing/invalid power: {skipped_missing_power} | Bad ts: {skipped_bad_ts}"
    )
    return total


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest energy readings from a CSV file")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--ts", dest="ts_col", help="Timestamp column name (override)")
    parser.add_argument("--machine", dest="machine_col", help="Machine ID column name (override)")
    parser.add_argument("--power", dest="power_col", help="Power (kW) column name (override)")
    parser.add_argument("--state", dest="state_col", help="State/Status column name (optional)")
    parser.add_argument("--throughput", dest="throughput_col", help="Throughput column name (optional; not stored)")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    inserted = ingest_csv(csv_path, args.ts_col, args.machine_col, args.power_col, args.state_col, args.throughput_col)
    print(f"Inserted {inserted} rows from {csv_path}")


if __name__ == "__main__":
    main()
