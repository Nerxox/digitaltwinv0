"""
Production readiness tests for Digital Twin Energy API.
"""
import pytest
import asyncio
import httpx
import json
from fastapi.testclient import TestClient
from backend_ai.main import app
from backend_ai.auth import create_access_token
from backend_ai.monitoring import HealthChecker

class TestProductionReadiness:
    """Test suite for production readiness."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers."""
        token = create_access_token(data={"sub": "admin", "scopes": ["read", "write", "admin"]})
        return {"Authorization": f"Bearer {token}"}
    
    def test_health_endpoints(self, client):
        """Test health check endpoints."""
        # Test liveness probe
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        
        # Test readiness probe
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
    
    def test_security_headers(self, client):
        """Test security headers are present."""
        response = client.get("/healthz")
        
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Referrer-Policy"
        ]
        
        for header in security_headers:
            assert header in response.headers
    
    def test_rate_limiting(self, client):
        """Test rate limiting functionality."""
        # Make multiple requests quickly
        responses = []
        for _ in range(105):  # Exceed rate limit of 100/min
            response = client.get("/api/v1/insights/machines")
            responses.append(response.status_code)
        
        # Should have some 429 responses
        assert 429 in responses
    
    def test_authentication_required(self, client):
        """Test that protected endpoints require authentication."""
        response = client.get("/api/v1/insights/machines")
        assert response.status_code == 401
    
    def test_authorization_scopes(self, client):
        """Test authorization scopes."""
        # Test with read-only user
        read_token = create_access_token(data={"sub": "viewer", "scopes": ["read"]})
        read_headers = {"Authorization": f"Bearer {read_token}"}
        
        # Should be able to read
        response = client.get("/api/v1/insights/machines", headers=read_headers)
        assert response.status_code in [200, 404]  # 404 if no data, but not 403
        
        # Should not be able to write
        response = client.post("/api/v1/predict", 
                              json={"machine_id": "M001", "horizon_min": 5, "algo": "lstm", "scope": "per_machine"},
                              headers=read_headers)
        assert response.status_code == 403
    
    def test_input_validation(self, client, auth_headers):
        """Test input validation."""
        # Invalid machine ID
        response = client.post("/api/v1/predict",
                             json={"machine_id": "INVALID@ID", "horizon_min": 5, "algo": "lstm", "scope": "per_machine"},
                             headers=auth_headers)
        assert response.status_code == 422
        
        # Invalid horizon
        response = client.post("/api/v1/predict",
                             json={"machine_id": "M001", "horizon_min": 2000, "algo": "lstm", "scope": "per_machine"},
                             headers=auth_headers)
        assert response.status_code == 422
        
        # Invalid algorithm
        response = client.post("/api/v1/predict",
                             json={"machine_id": "M001", "horizon_min": 5, "algo": "invalid", "scope": "per_machine"},
                             headers=auth_headers)
        assert response.status_code == 422
    
    def test_api_documentation(self, client):
        """Test API documentation is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
        
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "openapi" in response.json()
    
    def test_error_handling(self, client, auth_headers):
        """Test error handling."""
        # Test 404 for non-existent endpoint
        response = client.get("/api/v1/nonexistent", headers=auth_headers)
        assert response.status_code == 404
        
        # Test malformed JSON
        response = client.post("/api/v1/predict",
                             data="invalid json",
                             headers={**auth_headers, "Content-Type": "application/json"})
        assert response.status_code == 422
    
    def test_performance_metrics(self, client, auth_headers):
        """Test that performance metrics are collected."""
        # Make a request
        response = client.get("/api/v1/insights/machines", headers=auth_headers)
        
        # Check that request ID is present
        assert "X-Request-ID" in response.headers
    
    def test_system_health(self):
        """Test system health checker."""
        health = HealthChecker.get_system_health()
        
        assert "timestamp" in health
        assert "uptime_seconds" in health
        assert "database" in health
        assert "models" in health
        assert "system" in health
        
        # Check that system metrics are reasonable
        system = health["system"]
        assert 0 <= system["cpu_percent"] <= 100
        assert 0 <= system["memory_percent"] <= 100
        assert 0 <= system["disk_percent"] <= 100

class TestLoadTesting:
    """Load testing for production readiness."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        async def make_request():
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8000/healthz")
                return response.status_code
        
        # Make 50 concurrent requests
        tasks = [make_request() for _ in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        success_count = sum(1 for result in results if result == 200)
        assert success_count >= 45  # Allow for some failures under load
    
    def test_memory_usage(self, client, auth_headers):
        """Test memory usage doesn't grow excessively."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Make many requests
        for _ in range(100):
            response = client.get("/api/v1/insights/machines", headers=auth_headers)
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 50MB)
        assert memory_growth < 50 * 1024 * 1024

class TestDataIntegrity:
    """Test data integrity and consistency."""
    
    def test_database_connections(self):
        """Test database connection health."""
        health = HealthChecker.check_database()
        assert health["status"] == "healthy"
    
    def test_model_availability(self):
        """Test model availability."""
        health = HealthChecker.check_models()
        assert "models" in health
        assert "status" in health

if __name__ == "__main__":
    pytest.main([__file__, "-v"])