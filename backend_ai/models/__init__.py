"""Database models for the Digital Twin Energy application."""
# Import models to make them available when the package is imported
# Importing Base from a separate module to avoid circular imports
from .base import Base  # noqa: F401
from .energy_reading import EnergyReading  # noqa: F401

__all__ = ['Base', 'EnergyReading']
