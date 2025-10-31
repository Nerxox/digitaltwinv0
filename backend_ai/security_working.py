"""
Working security utilities for Digital Twin Energy API.
"""
import re
import time
from typing import Dict, Any, Optional
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer
import logging

logger = logging.getLogger(__name__)

# Rate limiting storage (use Redis in production)
rate_limit_storage: Dict[str, Dict[str, Any]] = {}

class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed for given IP."""
        now = time.time()
        minute_window = int(now // 60)
        
        if client_ip not in rate_limit_storage:
            rate_limit_storage[client_ip] = {}
        
        client_data = rate_limit_storage[client_ip]
        
        if minute_window not in client_data:
            client_data[minute_window] = 0
        
        # Clean old windows
        for window in list(client_data.keys()):
            if window < minute_window - 1:
                del client_data[window]
        
        if client_data[minute_window] >= self.requests_per_minute:
            return False
        
        client_data[minute_window] += 1
        return True

# Global rate limiter
rate_limiter = RateLimiter(requests_per_minute=100)

def get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    # Check for forwarded headers first
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct connection
    return request.client.host if request.client else "unknown"

def validate_machine_id(machine_id: str) -> bool:
    """Validate machine ID format."""
    # Allow alphanumeric with underscores and hyphens
    pattern = r'^[A-Za-z0-9_-]+$'
    return bool(re.match(pattern, machine_id)) and len(machine_id) <= 50

def validate_horizon(horizon_min: int) -> bool:
    """Validate prediction horizon."""
    return 1 <= horizon_min <= 1440  # 1 minute to 24 hours

def validate_algorithm(algo: str) -> bool:
    """Validate algorithm name."""
    allowed_algos = {"lstm", "gru", "xgboost", "rf", "lgb"}
    return algo in allowed_algos

def validate_scope(scope: str) -> bool:
    """Validate model scope."""
    allowed_scopes = {"per_machine", "global"}
    return scope in allowed_scopes

def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input."""
    if not text:
        return ""
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', text)
    
    # Limit length
    return sanitized[:max_length]

def check_rate_limit(request: Request) -> None:
    """Check rate limit for request."""
    client_ip = get_client_ip(request)
    
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )

class SecurityHeaders:
    """Add security headers to responses."""
    
    @staticmethod
    def add_security_headers(response):
        """Add security headers to response."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

def validate_prediction_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate prediction request data."""
    errors = []
    
    # Validate machine_id
    if "machine_id" in data:
        if isinstance(data["machine_id"], list):
            for mid in data["machine_id"]:
                if not validate_machine_id(str(mid)):
                    errors.append(f"Invalid machine_id format: {mid}")
        else:
            if not validate_machine_id(str(data["machine_id"])):
                errors.append(f"Invalid machine_id format: {data['machine_id']}")
    
    # Validate horizon
    if "horizon_min" in data:
        if isinstance(data["horizon_min"], list):
            for h in data["horizon_min"]:
                if not validate_horizon(int(h)):
                    errors.append(f"Invalid horizon_min: {h}")
        else:
            if not validate_horizon(int(data["horizon_min"])):
                errors.append(f"Invalid horizon_min: {data['horizon_min']}")
    
    # Validate algorithm
    if "algo" in data and not validate_algorithm(data["algo"]):
        errors.append(f"Invalid algorithm: {data['algo']}")
    
    # Validate scope
    if "scope" in data and not validate_scope(data["scope"]):
        errors.append(f"Invalid scope: {data['scope']}")
    
    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"validation_errors": errors}
        )
    
    return data