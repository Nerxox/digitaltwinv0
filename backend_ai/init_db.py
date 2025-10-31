"""Initialize the database with required tables and sample data."""
import sys
from pathlib import Path
import random
from datetime import datetime, timedelta

# Add the project root to the Python path
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend_ai.common.db import init_db, SessionLocal, engine
from backend_ai.models.energy_reading import EnergyReading

def create_tables():
    """Create all database tables."""
    print("Creating database tables...")
    init_db()
    print("[SUCCESS] Database tables created successfully")

def add_sample_data():
    """Add sample energy reading data to the database."""
    db = SessionLocal()
    try:
        # Check if we already have data
        existing = db.query(EnergyReading).first()
        if existing:
            print("[INFO] Sample data already exists, skipping...")
            return

        print("Adding sample data...")
        now = datetime.utcnow()
        
        # Add data for the last 24 hours (48 readings, every 30 minutes)
        for hours_ago in range(24, 0, -1):
            for minute in [0, 30]:  # Two readings per hour
                timestamp = now - timedelta(hours=hours_ago, minutes=minute)
                reading = EnergyReading(
                    machine_id="machine_001",
                    power_kw=round(random.uniform(10, 100), 2),  # Random power between 10-100 kW
                    ts=timestamp
                )
                db.add(reading)
        
        db.commit()
        print("[SUCCESS] Sample data added successfully")
        
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Error adding sample data: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    print("Initializing database...")
    create_tables()
    add_sample_data()
    print("\nDatabase initialization complete!")
