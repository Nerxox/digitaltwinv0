"""Database connection and session management using SQLAlchemy."""
import os
from typing import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Import Base from models to avoid circular imports
from backend_ai.models.base import Base

# Use SQLite for development
BASE_DIR = Path(__file__).resolve().parent.parent
SQLALCHEMY_DATABASE_URL = f"sqlite:///{BASE_DIR}/sql_app.db"

# Create SQLAlchemy engine with SQLite configuration
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,  # Enable connection health checks
    pool_recycle=300,    # Recycle connections after 5 minutes
    pool_size=5,         # Number of connections to keep open
    max_overflow=10,     # Max number of connections to create beyond pool_size
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency for getting database session.
    
    Yields:
        Session: Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    # Import models here to ensure they are registered with SQLAlchemy
    from backend_ai.models import energy_reading  # noqa: F401
    
    # Create all tables
    Base.metadata.create_all(bind=engine)


def test_connection() -> bool:
    """Test database connection.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"Database connection error: {e}")
        return False


if __name__ == "__main__":
    # Test database connection
    if test_connection():
        print("✅ Database connection successful")
    else:
        print("❌ Database connection failed")
