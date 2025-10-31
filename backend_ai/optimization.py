"""
Performance optimization utilities for Digital Twin Energy API.
"""
import asyncio
import time
import functools
from typing import Any, Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class ModelCache:
    """LRU cache for ML models to reduce loading time."""
    
    def __init__(self, max_size: int = 10):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get model from cache."""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, key: str, model: Any) -> None:
        """Put model in cache."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[key] = model
        self.access_times[key] = time.time()
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.access_times.clear()

# Global model cache
model_cache = ModelCache(max_size=20)

class PredictionBatcher:
    """Batch predictions to improve throughput."""
    
    def __init__(self, batch_size: int = 32, timeout: float = 0.1):
        self.batch_size = batch_size
        self.timeout = timeout
        self.pending_requests = []
        self.last_batch_time = time.time()
    
    async def add_request(self, request_data: Dict[str, Any]) -> Any:
        """Add request to batch."""
        self.pending_requests.append(request_data)
        
        # Process batch if full or timeout reached
        if (len(self.pending_requests) >= self.batch_size or 
            time.time() - self.last_batch_time > self.timeout):
            return await self._process_batch()
        
        return None
    
    async def _process_batch(self) -> List[Any]:
        """Process current batch."""
        if not self.pending_requests:
            return []
        
        batch = self.pending_requests.copy()
        self.pending_requests.clear()
        self.last_batch_time = time.time()
        
        # Process batch in parallel
        tasks = [self._process_single_request(req) for req in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def _process_single_request(self, request_data: Dict[str, Any]) -> Any:
        """Process single request within batch."""
        # This would be implemented based on your specific prediction logic
        # For now, return mock result
        return {"prediction": np.random.random(), "request_id": request_data.get("id")}

# Global prediction batcher
prediction_batcher = PredictionBatcher(batch_size=16, timeout=0.05)

class DatabaseOptimizer:
    """Database query optimization utilities."""
    
    @staticmethod
    def optimize_time_series_query(
        start_time: str, 
        end_time: str, 
        machine_id: str,
        limit: int = 1000
    ) -> str:
        """Generate optimized time series query."""
        return f"""
        SELECT timestamp, power_kw, machine_id
        FROM energy_readings 
        WHERE machine_id = '{machine_id}'
        AND timestamp BETWEEN '{start_time}' AND '{end_time}'
        ORDER BY timestamp DESC
        LIMIT {limit}
        """
    
    @staticmethod
    def create_indexes() -> List[str]:
        """Generate SQL for creating performance indexes."""
        return [
            "CREATE INDEX IF NOT EXISTS idx_energy_readings_machine_timestamp ON energy_readings(machine_id, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_energy_readings_timestamp ON energy_readings(timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_energy_readings_machine_id ON energy_readings(machine_id);",
        ]

class ModelOptimizer:
    """ML model optimization utilities."""
    
    @staticmethod
    def quantize_model(model_path: str, output_path: str) -> bool:
        """Quantize model to reduce size and improve inference speed."""
        try:
            import tensorflow as tf
            
            # Load model
            model = tf.keras.models.load_model(model_path)
            
            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Convert and save
            tflite_model = converter.convert()
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"Model quantized and saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Model quantization failed: {e}")
            return False
    
    @staticmethod
    def optimize_prediction_pipeline(
        data: np.ndarray,
        model: Any,
        batch_size: int = 32
    ) -> np.ndarray:
        """Optimize prediction pipeline with batching."""
        predictions = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_pred = model.predict(batch, verbose=0)
            predictions.extend(batch_pred)
        
        return np.array(predictions)

class AsyncOptimizer:
    """Async operation optimization utilities."""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """Run CPU-intensive function in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)
    
    def close(self):
        """Close thread pool executor."""
        self.executor.shutdown(wait=True)

# Global async optimizer
async_optimizer = AsyncOptimizer(max_workers=4)

def cache_result(ttl: int = 300):
    """Decorator to cache function results with TTL."""
    def decorator(func):
        cache = {}
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Check cache
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl:
                    return result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache[key] = (result, time.time())
            
            # Clean old entries
            current_time = time.time()
            cache = {k: v for k, v in cache.items() if current_time - v[1] < ttl}
            
            return result
        
        return wrapper
    return decorator

def measure_performance(func):
    """Decorator to measure function performance."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start_time
        
        logger.info(f"{func.__name__} executed in {duration:.3f}s")
        return result
    
    return wrapper

class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record performance metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            'value': value,
            'timestamp': time.time(),
            'tags': tags or {}
        })
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        summary = {}
        
        for name, values in self.metrics.items():
            if not values:
                continue
            
            recent_values = [v['value'] for v in values[-100:]]  # Last 100 values
            
            summary[name] = {
                'count': len(recent_values),
                'mean': np.mean(recent_values),
                'median': np.median(recent_values),
                'min': np.min(recent_values),
                'max': np.max(recent_values),
                'p95': np.percentile(recent_values, 95),
                'p99': np.percentile(recent_values, 99)
            }
        
        return summary

# Global performance monitor
performance_monitor = PerformanceMonitor()

class ConnectionPool:
    """Database connection pool for better performance."""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections = []
        self.available = []
        self.lock = asyncio.Lock()
    
    async def get_connection(self):
        """Get database connection from pool."""
        async with self.lock:
            if self.available:
                return self.available.pop()
            
            if len(self.connections) < self.max_connections:
                # Create new connection
                conn = await self._create_connection()
                self.connections.append(conn)
                return conn
            
            # Wait for connection to become available
            while not self.available:
                await asyncio.sleep(0.01)
            
            return self.available.pop()
    
    async def return_connection(self, conn):
        """Return connection to pool."""
        async with self.lock:
            self.available.append(conn)
    
    async def _create_connection(self):
        """Create new database connection."""
        # This would be implemented based on your database setup
        pass

# Global connection pool
connection_pool = ConnectionPool(max_connections=20)

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage."""
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to category if low cardinality
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        elif df[col].dtype == 'int64':
            # Downcast integers
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
                elif df[col].max() < 4294967295:
                    df[col] = df[col].astype('uint32')
        elif df[col].dtype == 'float64':
            # Downcast floats
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df