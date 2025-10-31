from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel
import time
import socket
from uuid import uuid4
from backend_ai.routers.predict import router as prediction_router
from backend_ai.routers.evaluate import router as evaluate_router
from backend_ai.routers.insights import router as insights_router
from backend_ai.routers.ann import router as ann_router
from backend_ai.routers.stations import router as stations_router
from backend_ai.services.best_config import load_winning_config
from backend_ai.common.api_models import Healthz, ApiError
from backend_ai.security_working import SecurityHeaders, check_rate_limit, get_client_ip
from backend_ai.auth_working import get_current_active_user
from backend_ai.monitoring_working import RequestLogger, metrics_collector

# Observability (Prometheus)
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    _INSTRUMENTATION_AVAILABLE = True
except Exception:
    _INSTRUMENTATION_AVAILABLE = False

from fastapi.openapi.utils import get_openapi

# Create FastAPI app instance first
app = FastAPI(title="DT Energy Backend")
# Global start time for uptime

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="DT Energy Backend",
        version="0.1.0",
        description="Digital Twin Energy Prediction System with FastAPI backend, SQLite database, and LSTM-based energy consumption forecasting.",
        routes=app.routes,
    )

    # Add custom schemas and examples
    # Merge our additional schemas into FastAPI-generated components
    components = openapi_schema.get("components", {})
    schemas = components.get("schemas", {})
    extra_schemas = {
            "Algo": {
                "type": "string",
                "enum": ["lstm", "gru", "xgboost", "rf"],
                "description": "Forecasting algorithm identifier."
            },
            "Scope": {
                "type": "string",
                "enum": ["per_machine", "global"],
                "description": "Model scope; per_machine typically routes to a machine-specific model."
            },
            "MachineId": {
                "oneOf": [
                    {"type": "string"},
                    {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                ],
                "description": "A single machine id (e.g., \"M001\") or a list [\"M001\",\"M002\"]."
            },
            "HorizonMin": {
                "oneOf": [
                    {"type": "integer", "minimum": 1},
                    {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1}
                    }
                ],
                "description": "Forecast horizon in minutes, or a list of horizons."
            },
            "PredictRequest": {
                "type": "object",
                "required": ["machine_id", "horizon_min", "algo", "scope"],
                "properties": {
                    "machine_id": {"$ref": "#/components/schemas/MachineId"},
                    "horizon_min": {"$ref": "#/components/schemas/HorizonMin"},
                    "algo": {"$ref": "#/components/schemas/Algo"},
                    "scope": {"$ref": "#/components/schemas/Scope"},
                    "want_quantiles": {
                        "type": "boolean",
                        "default": False,
                        "description": "If true, include predictive intervals from API-3 (when supported)."
                    },
                    "quantile_levels": {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                        "default": [0.1, 0.5, 0.9],
                        "description": "Quantiles to return when want_quantiles is true."
                    }
                }
            },
            "ForecastPoint": {
                "type": "object",
                "required": ["t", "yhat"],
                "properties": {
                    "t": {
                        "type": "string",
                        "format": "date-time",
                        "description": "ISO-8601 timestamp of the forecasted period end (UTC)."
                    },
                    "horizon_min": {
                        "type": "integer",
                        "description": "Horizon (minutes) for this point."
                    },
                    "yhat": {
                        "type": "number",
                        "description": "Predicted active power (W) or energy rate."
                    },
                    "drift_flag": {
                        "type": "boolean",
                        "description": "True if simple drift heuristics detect out-of-distribution context."
                    },
                    "quantiles": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": "Optional map { \"q0.1\": ..., \"q0.5\": ..., \"q0.9\": ... }."
                    }
                }
            },
            "MachineForecast": {
                "type": "object",
                "required": ["machine_id", "algo", "scope", "generated_at", "unit", "points"],
                "properties": {
                    "machine_id": {"type": "string"},
                    "algo": {"$ref": "#/components/schemas/Algo"},
                    "scope": {"$ref": "#/components/schemas/Scope"},
                    "horizons_available": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Horizons supported by the backing model at generation time."
                    },
                    "generated_at": {"type": "string", "format": "date-time"},
                    "unit": {
                        "type": "string",
                        "example": "W",
                        "description": "Unit of yhat."
                    },
                    "points": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/ForecastPoint"}
                    }
                }
            },
            "PredictResponse": {
                "oneOf": [
                    {
                        "type": "object",
                        "required": ["result"],
                        "properties": {
                            "result": {"$ref": "#/components/schemas/MachineForecast"}
                        }
                    },
                    {
                        "type": "object",
                        "required": ["result"],
                        "properties": {
                            "result": {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/MachineForecast"}
                            }
                        }
                    }
                ]
            },
            "ApiError": {
                "type": "object",
                "required": ["error", "code"],
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Stable, programmatic error code.",
                        "examples": ["VALIDATION_ERROR", "MODEL_NOT_READY", "HORIZON_UNSUPPORTED", "NOT_FOUND"]
                    },
                    "error": {
                        "type": "string",
                        "description": "Human-readable short message."
                    },
                    "details": {
                        "type": "object",
                        "additionalProperties": True,
                        "description": "Optional structured details (field errors, etc)."
                    }
                }
            },
            "ValidationError": {
                "type": "object",
                "required": ["loc", "msg", "type"],
                "properties": {
                    "loc": {
                        "title": "Location",
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "integer"}
                            ]
                        }
                    },
                    "msg": {"title": "Message", "type": "string"},
                    "type": {"title": "Error Type", "type": "string"}
                }
            },
            "HTTPValidationError": {
                "type": "object",
                "properties": {
                    "detail": {
                        "title": "Detail",
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/ValidationError"}
                    }
                }
            }
    }
    # Update and assign merged components
    schemas.update(extra_schemas)
    components["schemas"] = schemas
    openapi_schema["components"] = components

    # Add examples to paths
    if "paths" in openapi_schema and "/api/v1/predict" in openapi_schema["paths"]:
        predict_path = openapi_schema["paths"]["/api/v1/predict"]["post"]
        predict_path["requestBody"]["content"]["application/json"]["examples"] = {
            "single_machine_multi_horizon": {
                "value": {
                    "machine_id": "M001",
                    "horizon_min": [1, 5, 15, 30],
                    "algo": "lstm",
                    "scope": "per_machine"
                }
            },
            "batch_machines_single_horizon": {
                "value": {
                    "machine_id": ["M001", "M002"],
                    "horizon_min": 15,
                    "algo": "lstm",
                    "scope": "per_machine"
                }
            },
            "quantiles_on": {
                "value": {
                    "machine_id": "M001",
                    "horizon_min": [5, 15],
                    "algo": "lstm",
                    "scope": "per_machine",
                    "want_quantiles": True,
                    "quantile_levels": [0.1, 0.5, 0.9]
                }
            }
        }

        predict_path["responses"]["200"]["content"]["application/json"]["schema"] = {
            "$ref": "#/components/schemas/PredictResponse"
        }
        predict_path["responses"]["400"]["content"]["application/json"]["schema"] = {
            "$ref": "#/components/schemas/ApiError"
        }
        predict_path["responses"]["404"]["content"]["application/json"]["schema"] = {
            "$ref": "#/components/schemas/ApiError"
        }
        predict_path["responses"]["503"]["content"]["application/json"]["schema"] = {
            "$ref": "#/components/schemas/ApiError"
        }

    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Assign custom openapi function after app is defined
app.openapi = custom_openapi

_START_TIME = time.time()

# Security middleware with rate limiting and monitoring
@app.middleware("http")
async def add_request_context(request: Request, call_next):
    start_time = time.time()
    
    # Rate limiting for API endpoints
    if request.url.path.startswith("/api/"):
        check_rate_limit(request)
    
    response: Response = await call_next(request)
    duration = time.time() - start_time
    
    # X-Request-ID
    rid = request.headers.get("X-Request-ID") or str(uuid4())
    response.headers["X-Request-ID"] = rid
    
    # Add security headers
    SecurityHeaders.add_security_headers(response)
    
    # Cache-Control on health/readiness/predict
    if request.url.path in {"/healthz", "/readyz", "/api/v1/predict"}:
        response.headers["Cache-Control"] = "no-store"
    
    # Log request and collect metrics
    RequestLogger.log_request(request, response, duration)
    
    # Collect system metrics periodically
    if int(time.time()) % 60 == 0:  # Every minute
        metrics_collector.collect_system_metrics()
    
    return response

# Include routers (let each router define its own tags to avoid duplication)
app.include_router(prediction_router, prefix="/api/v1")
app.include_router(evaluate_router, prefix="/api/v1")
app.include_router(insights_router, prefix="/api/v1")
app.include_router(ann_router, prefix="/api/v1")
app.include_router(stations_router, prefix="/api/v1")

@app.get("/healthz", tags=["Observability"])
def healthz():
    """Liveness probe - returns 200 if the app loop is alive."""
    return {"status": "ok"}

@app.get("/readyz", tags=["Observability"])
def readyz(response: Response):
    """Readiness probe - returns 200 only when DB and model registry are ready."""
    from backend_ai.monitoring_working import HealthChecker
    
    health = HealthChecker.get_system_health()
    checks = {
        "db": health["database"]["status"],
        "models": health["models"]["status"],
        "system": "ok" if health["system"]["cpu_percent"] < 90 else "degraded"
    }

    all_ready = all(status in ["ok", "healthy"] for status in checks.values())

    if not all_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ApiError(
            code="NOT_READY",
            error="One or more dependencies are not ready.",
            details={"checks": checks}
        ).model_dump()

    return {
        "status": "ready",
        "checks": checks
    }

# Authentication endpoints
class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/v1/auth/login", tags=["Authentication"])
def login(login_data: LoginRequest):
    """Login endpoint for authentication."""
    from backend_ai.auth_working import authenticate_user, create_access_token
    
    user = authenticate_user(login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user["username"], "scopes": user["scopes"]}
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "username": user["username"],
            "email": user["email"],
            "full_name": user["full_name"],
            "scopes": user["scopes"]
        }
    }

@app.get("/api/v1/auth/me", tags=["Authentication"])
def get_current_user_info(current_user = Depends(get_current_active_user)):
    """Get current user information."""
    return {
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "scopes": current_user.scopes,
        "is_active": current_user.is_active
    }

# Load best model configuration at startup
load_winning_config()

# Expose /metrics if instrumentator is available
try:
    _avail = _INSTRUMENTATION_AVAILABLE
except Exception:
    _avail = False
if _avail:
    Instrumentator().instrument(app).expose(app, include_in_schema=False)