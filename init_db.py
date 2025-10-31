from backend_ai.common.db import engine, Base
from backend_ai.models.energy_reading import EnergyReading

def init_db():
    """Initialize the database by creating all tables."""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

if __name__ == "__main__":
    init_db()
