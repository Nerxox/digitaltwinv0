import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

# Ensure project root is on path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend_ai.common.db import SessionLocal, init_db
from backend_ai.models.energy_reading import EnergyReading


def main(machine_id: str = "machine_001", points: int = 24):
    init_db()
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        rows = []
        # Insert hourly points going back in time so we have chronological series
        for i in range(points, 0, -1):
            ts = now - timedelta(hours=i)
            rows.append(EnergyReading(
                machine_id=machine_id,
                power_kw=round(random.uniform(10, 100), 2),
                ts=ts,
                status="ok"
            ))
        db.add_all(rows)
        db.commit()
        print(f"Seeded {points} records for {machine_id}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
