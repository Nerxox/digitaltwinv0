"""
Load testing suite for Digital Twin Energy API.
"""
import asyncio
import aiohttp
import time
import statistics
import json
from typing import List, Dict, Any
import logging
from dataclasses import dataclass
import argparse

logger = logging.getLogger(__name__)

@dataclass
class LoadTestResult:
    """Load test result data."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    requests_per_second: float
    average_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_rate: float

class LoadTester:
    """Comprehensive load testing for Digital Twin Energy API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.results = []
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def make_request(self, endpoint: str, method: str = "GET", 
                          data: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Make a single HTTP request."""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                async with self.session.get(url, headers=headers) as response:
                    response_time = time.time() - start_time
                    return {
                        "status_code": response.status,
                        "response_time": response_time,
                        "success": 200 <= response.status < 300,
                        "content_length": len(await response.text())
                    }
            elif method.upper() == "POST":
                async with self.session.post(url, json=data, headers=headers) as response:
                    response_time = time.time() - start_time
                    return {
                        "status_code": response.status,
                        "response_time": response_time,
                        "success": 200 <= response.status < 300,
                        "content_length": len(await response.text())
                    }
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "status_code": 0,
                "response_time": response_time,
                "success": False,
                "error": str(e)
            }
    
    async def run_concurrent_requests(self, endpoint: str, num_requests: int, 
                                    concurrency: int = 10, method: str = "GET",
                                    data: Dict[str, Any] = None, headers: Dict[str, str] = None) -> LoadTestResult:
        """Run concurrent requests to test load."""
        semaphore = asyncio.Semaphore(concurrency)
        results = []
        
        async def make_request_with_semaphore():
            async with semaphore:
                return await self.make_request(endpoint, method, data, headers)
        
        start_time = time.time()
        
        # Create tasks
        tasks = [make_request_with_semaphore() for _ in range(num_requests)]
        
        # Execute all requests
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Process results
        successful_requests = sum(1 for r in results if isinstance(r, dict) and r.get("success", False))
        failed_requests = num_requests - successful_requests
        
        response_times = [r.get("response_time", 0) for r in results if isinstance(r, dict)]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p95_response_time = self._percentile(response_times, 95)
            p99_response_time = self._percentile(response_times, 99)
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p95_response_time = p99_response_time = 0
        
        return LoadTestResult(
            total_requests=num_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            requests_per_second=num_requests / total_time if total_time > 0 else 0,
            average_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            error_rate=failed_requests / num_requests if num_requests > 0 else 0
        )
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    async def test_health_endpoints(self) -> Dict[str, LoadTestResult]:
        """Test health check endpoints."""
        results = {}
        
        # Test /healthz
        logger.info("Testing /healthz endpoint...")
        results["healthz"] = await self.run_concurrent_requests("/healthz", 100, 20)
        
        # Test /readyz
        logger.info("Testing /readyz endpoint...")
        results["readyz"] = await self.run_concurrent_requests("/readyz", 100, 20)
        
        return results
    
    async def test_api_endpoints(self, auth_token: str = None) -> Dict[str, LoadTestResult]:
        """Test API endpoints with authentication."""
        results = {}
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        
        # Test insights endpoint
        logger.info("Testing /api/v1/insights/machines endpoint...")
        results["insights"] = await self.run_concurrent_requests("/api/v1/insights/machines", 50, 10, headers=headers)
        
        # Test prediction endpoint
        logger.info("Testing /api/v1/predict endpoint...")
        prediction_data = {
            "machine_id": "M001",
            "horizon_min": 15,
            "algo": "lstm",
            "scope": "per_machine"
        }
        results["predict"] = await self.run_concurrent_requests("/api/v1/predict", 30, 5, "POST", prediction_data, headers)
        
        return results
    
    async def test_authentication(self) -> LoadTestResult:
        """Test authentication endpoint."""
        logger.info("Testing authentication...")
        
        # Test login endpoint (if available)
        login_data = {
            "username": "admin",
            "password": "admin123"
        }
        
        return await self.run_concurrent_requests("/api/v1/auth/login", 50, 10, "POST", login_data)
    
    async def test_rate_limiting(self) -> LoadTestResult:
        """Test rate limiting functionality."""
        logger.info("Testing rate limiting...")
        
        # Make many requests quickly to trigger rate limiting
        return await self.run_concurrent_requests("/api/v1/insights/machines", 200, 50)
    
    async def test_memory_usage(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Test memory usage over time."""
        logger.info(f"Testing memory usage for {duration_seconds} seconds...")
        
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < duration_seconds:
            await self.make_request("/healthz")
            request_count += 1
            await asyncio.sleep(0.1)  # 10 requests per second
        
        return {
            "duration": duration_seconds,
            "requests_made": request_count,
            "requests_per_second": request_count / duration_seconds
        }
    
    async def run_comprehensive_test(self, auth_token: str = None) -> Dict[str, Any]:
        """Run comprehensive load test suite."""
        logger.info("Starting comprehensive load test...")
        
        results = {
            "timestamp": time.time(),
            "base_url": self.base_url,
            "tests": {}
        }
        
        # Health endpoint tests
        results["tests"]["health"] = await self.test_health_endpoints()
        
        # API endpoint tests
        if auth_token:
            results["tests"]["api"] = await self.test_api_endpoints(auth_token)
        
        # Authentication test
        results["tests"]["auth"] = await self.test_authentication()
        
        # Rate limiting test
        results["tests"]["rate_limiting"] = await self.test_rate_limiting()
        
        # Memory usage test
        results["tests"]["memory"] = await self.test_memory_usage(30)
        
        return results

def print_results(results: Dict[str, Any]):
    """Print load test results in a formatted way."""
    print("\n" + "="*80)
    print("LOAD TEST RESULTS")
    print("="*80)
    
    for test_category, test_results in results["tests"].items():
        print(f"\n{test_category.upper()} TESTS:")
        print("-" * 40)
        
        if isinstance(test_results, dict):
            for test_name, result in test_results.items():
                if isinstance(result, LoadTestResult):
                    print(f"\n{test_name}:")
                    print(f"  Total Requests: {result.total_requests}")
                    print(f"  Successful: {result.successful_requests}")
                    print(f"  Failed: {result.failed_requests}")
                    print(f"  Error Rate: {result.error_rate:.2%}")
                    print(f"  Requests/sec: {result.requests_per_second:.2f}")
                    print(f"  Avg Response Time: {result.average_response_time:.3f}s")
                    print(f"  Min Response Time: {result.min_response_time:.3f}s")
                    print(f"  Max Response Time: {result.max_response_time:.3f}s")
                    print(f"  P95 Response Time: {result.p95_response_time:.3f}s")
                    print(f"  P99 Response Time: {result.p99_response_time:.3f}s")
                else:
                    print(f"\n{test_name}: {result}")
        else:
            print(f"  {test_results}")

async def main():
    """Main load testing function."""
    parser = argparse.ArgumentParser(description="Load test Digital Twin Energy API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--auth-token", help="Authentication token for protected endpoints")
    parser.add_argument("--output", help="Output file for results (JSON format)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    async with LoadTester(args.url) as tester:
        results = await tester.run_comprehensive_test(args.auth_token)
        
        # Print results
        print_results(results)
        
        # Save results to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())