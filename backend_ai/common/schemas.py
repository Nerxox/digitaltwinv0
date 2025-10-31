from pydantic import BaseModel, Field, StringConstraints, validator
from typing import List, Optional, Literal, Any, Dict, Union
from datetime import datetime
from enum import Enum

# Define the MachineStatus enum
class MachineStatus(str, Enum):
    RUN = "RUN"
    IDLE = "IDLE"
    STOP = "STOP"

class EnergyReadingBase(BaseModel):
    """Base model for energy reading data."""
    machine_id: str = Field(..., min_length=1, max_length=50, description="Unique identifier for the machine")
    ts: datetime = Field(..., description="Timestamp of the reading in UTC")
    power_kw: float = Field(..., ge=0, description="Power consumption in kilowatts")
    status: MachineStatus = Field(..., description="Current status of the machine")

class EnergyReadingCreate(EnergyReadingBase):
    """Model for creating a new energy reading (from MQTT)."""
    pass

class EnergyReading(EnergyReadingBase):
    """Complete energy reading model including database ID."""
    id: int = Field(..., description="Unique identifier for the reading")

    class Config:
        """Pydantic configuration."""
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MachineStats(BaseModel):
    """Statistics for a machine over a time period."""
    machine_id: str
    time_bucket: datetime
    avg_power_kw: float
    min_power_kw: float
    max_power_kw: float
    total_energy_kwh: float

class TimeRangeStats(BaseModel):
    """Statistics for multiple machines over a time range."""
    start_time: datetime
    end_time: datetime
    stats: List[MachineStats]

class MachineStatusCount(BaseModel):
    """Count of readings by status for a machine."""
    status: MachineStatus
    count: int

class MachineSummary(BaseModel):
    """Summary statistics for a machine."""
    machine_id: str
    current_power_kw: float
    current_status: MachineStatus
    uptime_percentage: float
    avg_power_kw: float
    status_counts: List[MachineStatusCount]

# Request models
class TimeRangeQuery(BaseModel):
    """Query parameters for time-based queries."""
    start_time: datetime
    end_time: datetime
    machine_ids: Optional[List[str]] = None

    @validator('end_time')
    def validate_time_range(cls, v, values):
        """Validate that end_time is after start_time."""
        if 'start_time' in values and v < values['start_time']:
            raise ValueError('end_time must be after start_time')
        return v

# Response models
class HealthCheck(BaseModel):
    """Health check response model."""
    status: str
    database: bool = Field(..., description="Database connection status")
    version: str = Field(..., description="API version")

class ErrorResponse(BaseModel):
    """Standard error response model."""
    detail: str = Field(..., description="Error message")
    code: int = Field(..., description="HTTP status code")
    error_type: str = Field(..., description="Type of error")