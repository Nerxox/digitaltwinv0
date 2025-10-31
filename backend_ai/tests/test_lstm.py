import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import numpy as np
from backend_ai.models.ml.train_lstm import LSTMForecaster

def test_sequence_creation():
    """Test that sequences are created with correct shapes."""
    # Create test data
    data = np.arange(100).reshape(-1, 1)
    lookback = 10  # Define lookback
    horizon = 5
    
    forecaster = LSTMForecaster(lookback=lookback, horizon=horizon)
    X, y = forecaster.create_sequences(data)
    
    # Check shapes
    assert X.shape == (100 - lookback - horizon + 1, lookback)
    assert y.shape == (100 - lookback - horizon + 1, horizon)
    
    # Check sequence alignment
    for i in range(len(X)):
        assert np.array_equal(X[i], data[i:i+lookback].flatten())
        assert np.array_equal(y[i], data[i+lookback:i+lookback+horizon].flatten())

def test_mape_calculation():
    """Test MAPE calculation."""
    forecaster = LSTMForecaster()
    
    # Test with perfect prediction
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    mae, mape = forecaster.evaluate(y_true, y_pred)
    assert mae == 0.0
    assert mape == 0.0
    
    # Test with some error
    y_pred = np.array([1.1, 1.9, 3.1])
    mae, mape = forecaster.evaluate(y_true, y_pred)
    assert mae == pytest.approx(0.1, abs=1e-2)
    assert mape > 0.0