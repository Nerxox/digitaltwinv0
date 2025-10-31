"""SQLAlchemy models for energy readings."""
from datetime import datetime
from sqlalchemy import Column, DateTime, Float, Integer, String, func
from sqlalchemy.dialects.postgresql import TIMESTAMP

# Import Base from the base module to avoid circular imports
from .base import Base


class EnergyReading(Base):
    """Database model for energy readings from machines.
    
    This table is optimized for time-series data and uses TimescaleDB's
    hypertable for better performance with time-series data.
    """
    __tablename__ = "energy_readings"
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String(50), nullable=False, index=True, comment="Machine identifier")
    ts = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        index=True,
        server_default=func.now(),
        comment="Timestamp in UTC"
    )
    power_kw = Column(
        Float,
        nullable=False,
        comment="Power consumption in kilowatts"
    )
    status = Column(
        String(10),
        nullable=False,
        index=True,
        comment="Machine status (RUN, IDLE, STOP)"
    )
    
    # Add a composite index on (machine_id, ts) for common query patterns
    __table_args__ = (
        {
            "postgresql_using": "btree",
            "comment": "Table for storing machine energy readings with time-series optimization"
        },
    )
    
    def __repr__(self):
        return (
            f"<EnergyReading(id={self.id}, "
            f"machine_id='{self.machine_id}', "
            f"ts='{self.ts}', "
            f"power_kw={self.power_kw}, "
            f"status='{self.status}')>"
        )
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "machine_id": self.machine_id,
            "ts": self.ts.isoformat() if self.ts else None,
            "power_kw": self.power_kw,
            "status": self.status
        }
