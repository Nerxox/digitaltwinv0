import random
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy.orm import Session

from backend_ai.common.db import SessionLocal, engine
from backend_ai.models.energy_reading import EnergyReading

def add_sample_data():
    """Add sample data to the database for testing."""
    db = SessionLocal()
    
    try:
        # Generate sample data for the last 24 hours
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)
        
        # Generate timestamps at 1-minute intervals
        timestamps = [start_time + timedelta(minutes=i) for i in range(24 * 60)]
        
        # Generate power values with some realistic patterns
        base_load = 10.0  # Base load in kW
        peak_load = 50.0   # Peak load in kW
        
        for i, ts in enumerate(timestamps):
            # Add some daily variation (higher during the day, lower at night)
            hour = ts.hour
            if 8 <= hour < 20:  # Daytime
                variation = np.sin((hour - 8) * np.pi / 12) * 0.8 + 0.2  # 0.2 to 1.0
            else:  # Nighttime
                variation = 0.2  # 20% of peak at night
                
            # Add some random noise
            noise = random.uniform(-5, 5)
            
            # Calculate power value
            power_kw = base_load + (peak_load - base_load) * variation + noise
            power_kw = max(0, power_kw)  # Ensure power is not negative
            
            # Create a new reading
            reading = EnergyReading(
                machine_id="M1",
                ts=ts,
                power_kw=power_kw,
                status="RUN" if power_kw > 15 else "IDLE"
            )
            
            db.add(reading)
        
        # Commit the changes
        db.commit()
        print(f"Added {len(timestamps)} sample readings to the database.")
        
    except Exception as e:
        print(f"Error adding sample data: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    # Make sure the tables exist
    from backend_ai.common.db import Base
    Base.metadata.create_all(bind=engine)
    
    # Add sample data
    add_sample_data()
