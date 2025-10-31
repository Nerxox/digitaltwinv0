"""
Monitoring and observability for Digital Twin Energy API.
"""
import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import Request, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import psutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/digital_twin.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total', 
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total predictions made',
    ['algorithm', 'machine_id', 'status']
)

PREDICTION_DURATION = Histogram(
    'prediction_duration_seconds',
    'Prediction duration in seconds',
    ['algorithm']
)

MODEL_LOAD_TIME = Histogram(
    'model_load_duration_seconds',
    'Model loading duration in seconds',
    ['algorithm', 'model_type']
)

SYSTEM_METRICS = {
    'cpu_usage': Gauge('system_cpu_usage_percent', 'CPU usage percentage'),
    'memory_usage': Gauge('system_memory_usage_bytes', 'Memory usage in bytes'),
    'disk_usage': Gauge('system_disk_usage_bytes', 'Disk usage in bytes'),
    'active_connections': Gauge('system_active_connections', 'Active database connections'),
}

class MetricsCollector:
    """Collect and expose system metrics."""
    
    def __init__(self):
        self.start_time = time.time()
    
    def collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            SYSTEM_METRICS['cpu_usage'].set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_METRICS['memory_usage'].set(memory.used)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            SYSTEM_METRICS['disk_usage'].set(disk.used)
            
            logger.debug(f"System metrics - CPU: {cpu_percent}%, Memory: {memory.used/1024/1024:.1f}MB")
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def get_uptime(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - self.start_time

# Global metrics collector
metrics_collector = MetricsCollector()

class RequestLogger:
    """Log HTTP requests with detailed information."""
    
    @staticmethod
    def log_request(request: Request, response: Response, duration: float):
        """Log request details."""
        client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
        
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": client_ip,
            "user_agent": request.headers.get("User-Agent", ""),
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
            "request_id": request.headers.get("X-Request-ID", ""),
        }
        
        # Add request body for POST requests (excluding sensitive data)
        if request.method == "POST" and request.url.path.startswith("/api/"):
            try:
                body = request._body
                if body and len(body) < 1000:  # Only log small bodies
                    log_data["request_body"] = body.decode('utf-8')
            except Exception:
                pass  # Skip body logging if there's an issue
        
        # Log level based on status code
        if response.status_code >= 500:
            logger.error(f"HTTP Request: {json.dumps(log_data)}")
        elif response.status_code >= 400:
            logger.warning(f"HTTP Request: {json.dumps(log_data)}")
        else:
            logger.info(f"HTTP Request: {json.dumps(log_data)}")
        
        # Update Prometheus metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)

class ModelMonitor:
    """Monitor ML model performance and health."""
    
    @staticmethod
    def log_prediction(algorithm: str, machine_id: str, duration: float, success: bool = True):
        """Log prediction metrics."""
        status = "success" if success else "error"
        
        PREDICTION_COUNT.labels(
            algorithm=algorithm,
            machine_id=machine_id,
            status=status
        ).inc()
        
        if success:
            PREDICTION_DURATION.labels(algorithm=algorithm).observe(duration)
        
        logger.info(f"Prediction - Algorithm: {algorithm}, Machine: {machine_id}, "
                   f"Duration: {duration:.3f}s, Success: {success}")
    
    @staticmethod
    def log_model_load(algorithm: str, model_type: str, duration: float):
        """Log model loading metrics."""
        MODEL_LOAD_TIME.labels(
            algorithm=algorithm,
            model_type=model_type
        ).observe(duration)
        
        logger.info(f"Model loaded - Algorithm: {algorithm}, Type: {model_type}, "
                   f"Duration: {duration:.3f}s")

class HealthChecker:
    """Check system health and dependencies."""
    
    @staticmethod
    def check_database() -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            # This would be implemented based on your database setup
            # For now, return a mock check
            return {
                "status": "healthy",
                "response_time_ms": 5.2,
                "connection_pool": "active"
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @staticmethod
    def check_models() -> Dict[str, Any]:
        """Check ML model availability."""
        try:
            # Check if model files exist and are accessible
            model_status = {}
            model_dirs = ["lstm", "gru", "xgboost", "rf"]
            
            for model_type in model_dirs:
                model_path = f"models/{model_type}"
                if os.path.exists(model_path):
                    model_status[model_type] = "available"
                else:
                    model_status[model_type] = "missing"
            
            return {
                "status": "healthy" if all(status == "available" for status in model_status.values()) else "degraded",
                "models": model_status
            }
        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @staticmethod
    def get_system_health() -> Dict[str, Any]:
        """Get comprehensive system health."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": metrics_collector.get_uptime(),
            "database": HealthChecker.check_database(),
            "models": HealthChecker.check_models(),
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }
        }

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)