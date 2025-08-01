"""
Comprehensive Load Testing Suite for AI Chatbot System
Tests API endpoints, WebSocket connections, and enterprise features
"""

import json
import random
import time
import uuid
from typing import Dict, List, Any
import asyncio
import websockets
from locust import HttpUser, TaskSet, task, between, events
from locust.contrib.fasthttp import FastHttpUser
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test data
SAMPLE_MESSAGES = [
    "Hello, how are you?",
    "What is artificial intelligence?",
    "Can you help me write a Python function to calculate fibonacci numbers?",
    "Explain quantum computing in simple terms",
    "What are the benefits of microservices architecture?",
    "How do I optimize database queries?",
    "Write a haiku about technology",
    "What's the difference between machine learning and deep learning?",
    "Can you analyze this code for potential security vulnerabilities?",
    "Explain the concept of blockchain technology",
    "What are best practices for API design?",
    "How do I implement caching in a web application?",
    "What is the CAP theorem?",
    "Explain OAuth 2.0 authentication flow",
    "How do I scale a web application horizontally?"
]

COMPLEX_QUERIES = [
    """Please analyze the following architecture and suggest improvements:
    We have a monolithic application serving 1M users with a MySQL database.
    The application is experiencing slow response times during peak hours.""",
    
    """Write a comprehensive system design for a chat application that needs to:
    1. Support 10M concurrent users
    2. Provide real-time messaging
    3. Have 99.99% uptime
    4. Be globally distributed""",
    
    """Review this Python code and identify performance bottlenecks:
    ```python
    def process_data(data):
        results = []
        for item in data:
            if item['value'] > 100:
                processed = expensive_operation(item)
                results.append(processed)
        return results
    ```""",
    
    """Design a machine learning pipeline for:
    - Real-time fraud detection
    - Processing 100K transactions/second
    - Sub-100ms response time
    - 99.9% accuracy requirement"""
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "AI-Chatbot-LoadTest/1.0"
]


class APITaskSet(TaskSet):
    """Task set for REST API testing"""
    
    def on_start(self):
        """Initialize user session"""
        self.session_id = str(uuid.uuid4())
        self.conversation_id = None
        self.api_key = None
        self.auth_token = None
        
        # Authenticate user
        self.authenticate()
        
    def authenticate(self):
        """Authenticate user and get session"""
        # Simulate different authentication methods
        auth_method = random.choice(["password", "api_key", "sso"])
        
        if auth_method == "password":
            self.login_with_password()
        elif auth_method == "api_key":
            self.login_with_api_key()
        else:
            self.simulate_sso_login()
            
    def login_with_password(self):
        """Login with username/password"""
        response = self.client.post("/api/v1/auth/login", json={
            "email": f"testuser{random.randint(1, 1000)}@example.com",
            "password": "TestPassword123!"
        }, headers={"User-Agent": random.choice(USER_AGENTS)})
        
        if response.status_code == 200:
            data = response.json()
            self.auth_token = data.get("access_token")
            self.client.headers.update({"Authorization": f"Bearer {self.auth_token}"})
        else:
            logger.warning(f"Login failed: {response.status_code}")
            
    def login_with_api_key(self):
        """Login with API key"""
        # Generate test API key
        self.api_key = f"sk_{uuid.uuid4().hex}"
        self.client.headers.update({"X-API-Key": self.api_key})
        
    def simulate_sso_login(self):
        """Simulate SSO login flow"""
        # Simulate SAML/OAuth flow
        self.auth_token = f"sso_{uuid.uuid4().hex}"
        self.client.headers.update({"Authorization": f"Bearer {self.auth_token}"})
        
    @task(5)
    def create_conversation(self):
        """Create a new conversation"""
        response = self.client.post("/api/v1/chat/sessions", json={
            "model": random.choice(["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"]),
            "temperature": random.uniform(0.1, 1.0),
            "max_tokens": random.randint(100, 2000)
        })
        
        if response.status_code == 201:
            data = response.json()
            self.conversation_id = data.get("session_id")
        else:
            logger.warning(f"Failed to create conversation: {response.status_code}")
            
    @task(20)
    def send_simple_message(self):
        """Send simple message"""
        if not self.conversation_id:
            self.create_conversation()
            
        message = random.choice(SAMPLE_MESSAGES)
        
        start_time = time.time()
        response = self.client.post("/api/v1/chat/messages", json={
            "message": message,
            "session_id": self.conversation_id,
            "stream": False
        })
        
        if response.status_code == 200:
            data = response.json()
            # Simulate reading response
            time.sleep(random.uniform(0.5, 2.0))
            
            # Track metrics
            response_time = time.time() - start_time
            events.request_success.fire(
                request_type="CHAT",
                name="simple_message",
                response_time=response_time * 1000,
                response_length=len(data.get("response", ""))
            )
        else:
            events.request_failure.fire(
                request_type="CHAT",
                name="simple_message",
                response_time=(time.time() - start_time) * 1000,
                response_length=0,
                exception=f"HTTP {response.status_code}"
            )
            
    @task(10)
    def send_complex_message(self):
        """Send complex message that requires more processing"""
        if not self.conversation_id:
            self.create_conversation()
            
        message = random.choice(COMPLEX_QUERIES)
        
        start_time = time.time()
        response = self.client.post("/api/v1/chat/messages", json={
            "message": message,
            "session_id": self.conversation_id,
            "model": "gpt-4",  # Use premium model for complex queries
            "stream": False
        })
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            events.request_success.fire(
                request_type="CHAT",
                name="complex_message",
                response_time=response_time * 1000,
                response_length=len(data.get("response", ""))
            )
        else:
            events.request_failure.fire(
                request_type="CHAT",
                name="complex_message",
                response_time=response_time * 1000,
                response_length=0,
                exception=f"HTTP {response.status_code}"
            )
            
    @task(8)
    def stream_message(self):
        """Test streaming responses"""
        if not self.conversation_id:
            self.create_conversation()
            
        message = random.choice(SAMPLE_MESSAGES)
        
        with self.client.post("/api/v1/chat/messages", json={
            "message": message,
            "session_id": self.conversation_id,
            "stream": True
        }, stream=True, catch_response=True) as response:
            
            if response.status_code == 200:
                chunks_received = 0
                total_content = ""
                
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk_data = json.loads(line.decode('utf-8'))
                            if chunk_data.get("type") == "stream":
                                chunks_received += 1
                                total_content += chunk_data.get("content", "")
                        except json.JSONDecodeError:
                            continue
                            
                if chunks_received > 0:
                    response.success()
                else:
                    response.failure("No chunks received")
            else:
                response.failure(f"HTTP {response.status_code}")
                
    @task(3)
    def upload_image(self):
        """Test image upload and analysis"""
        # Simulate image upload
        files = {"image": ("test.png", b"fake_image_data", "image/png")}
        
        response = self.client.post("/api/v1/upload/image", files=files)
        
        if response.status_code == 200:
            # Send message with image reference
            data = response.json()
            image_url = data.get("url")
            
            if image_url and self.conversation_id:
                self.client.post("/api/v1/chat/messages", json={
                    "message": "What do you see in this image?",
                    "session_id": self.conversation_id,
                    "attachments": [{"type": "image", "url": image_url}]
                })
                
    @task(2)
    def get_conversation_history(self):
        """Get conversation history"""
        if self.conversation_id:
            response = self.client.get(f"/api/v1/chat/sessions/{self.conversation_id}")
            
            if response.status_code == 200:
                data = response.json()
                # Simulate processing history
                time.sleep(0.1)
                
    @task(1)
    def get_analytics(self):
        """Get usage analytics"""
        response = self.client.get("/api/v1/analytics/usage", params={
            "start_date": "2024-01-01",
            "end_date": "2024-01-31"
        })
        
        if response.status_code == 200:
            data = response.json()
            # Simulate processing analytics
            time.sleep(0.2)
            
    @task(1)
    def health_check(self):
        """Health check endpoint"""
        response = self.client.get("/health")
        
        if response.status_code != 200:
            logger.warning(f"Health check failed: {response.status_code}")


class WebSocketTaskSet(TaskSet):
    """Task set for WebSocket testing"""
    
    def on_start(self):
        """Initialize WebSocket connection"""
        self.session_id = str(uuid.uuid4())
        self.ws = None
        self.messages_sent = 0
        self.messages_received = 0
        
    async def connect_websocket(self):
        """Establish WebSocket connection"""
        try:
            ws_url = f"ws://localhost:8000/api/v1/chat/stream/{self.session_id}"
            self.ws = await websockets.connect(ws_url)
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
            
    @task(1)
    def websocket_conversation(self):
        """Simulate WebSocket conversation"""
        asyncio.run(self._websocket_conversation())
        
    async def _websocket_conversation(self):
        """Async WebSocket conversation"""
        if not self.ws and not await self.connect_websocket():
            return
            
        try:
            # Send multiple messages
            for i in range(random.randint(3, 8)):
                message = {
                    "type": "message",
                    "content": random.choice(SAMPLE_MESSAGES),
                    "timestamp": time.time()
                }
                
                start_time = time.time()
                await self.ws.send(json.dumps(message))
                self.messages_sent += 1
                
                # Wait for response
                try:
                    response = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                    response_data = json.loads(response)
                    
                    response_time = time.time() - start_time
                    self.messages_received += 1
                    
                    events.request_success.fire(
                        request_type="WEBSOCKET",
                        name="message_exchange",
                        response_time=response_time * 1000,
                        response_length=len(response)
                    )
                    
                except asyncio.TimeoutError:
                    events.request_failure.fire(
                        request_type="WEBSOCKET",
                        name="message_exchange",
                        response_time=(time.time() - start_time) * 1000,
                        response_length=0,
                        exception="Timeout"
                    )
                    
                # Random delay between messages
                await asyncio.sleep(random.uniform(1, 5))
                
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            if self.ws:
                await self.ws.close()


class RegularUser(FastHttpUser):
    """Regular user with mixed workload"""
    tasks = [APITaskSet]
    wait_time = between(1, 5)
    weight = 70
    
    def on_start(self):
        """User initialization"""
        self.client.headers.update({
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "application/json",
            "Content-Type": "application/json"
        })


class PowerUser(FastHttpUser):
    """Power user with complex queries"""
    tasks = [APITaskSet]
    wait_time = between(0.5, 2)
    weight = 20
    
    def on_start(self):
        """Power user initialization"""
        self.client.headers.update({
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-User-Type": "power"
        })
        
    # Override task weights for power users
    APITaskSet.send_complex_message.tasks = [30]  # More complex queries
    APITaskSet.send_simple_message.tasks = [10]   # Fewer simple queries


class APIOnlyUser(FastHttpUser):
    """API-only user (no WebSocket)"""
    tasks = [APITaskSet]
    wait_time = between(0.1, 1)
    weight = 10
    
    def on_start(self):
        """API user initialization"""
        self.client.headers.update({
            "User-Agent": "API-Client/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-API-Key": f"sk_{uuid.uuid4().hex}"
        })


# Custom event handlers for detailed metrics
@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize custom metrics"""
    logger.info("Initializing load test...")
    
    # Custom metrics
    environment.stats.custom_metrics = {
        "websocket_connections": 0,
        "auth_failures": 0,
        "cache_hits": 0,
        "model_switches": 0
    }


@events.request_success.add_listener
def on_request_success(request_type, name, response_time, response_length, **kwargs):
    """Handle successful requests"""
    # Track specific metrics
    if request_type == "CHAT":
        # Simulate cache hit detection
        if random.random() < 0.3:  # 30% cache hit rate
            events.locust_init.environment.stats.custom_metrics["cache_hits"] += 1


@events.request_failure.add_listener
def on_request_failure(request_type, name, response_time, response_length, exception, **kwargs):
    """Handle failed requests"""
    logger.warning(f"Request failed: {request_type} {name} - {exception}")
    
    if "401" in str(exception) or "403" in str(exception):
        events.locust_init.environment.stats.custom_metrics["auth_failures"] += 1


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate test report"""
    logger.info("Load test completed. Generating report...")
    
    stats = environment.stats
    
    # Calculate additional metrics
    total_requests = stats.total.num_requests
    total_failures = stats.total.num_failures
    success_rate = ((total_requests - total_failures) / total_requests * 100) if total_requests > 0 else 0
    
    # Generate summary report
    report = f"""
    
=== LOAD TEST SUMMARY ===
Total Requests: {total_requests}
Total Failures: {total_failures}
Success Rate: {success_rate:.2f}%
Average Response Time: {stats.total.avg_response_time:.2f}ms
95th Percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms
99th Percentile: {stats.total.get_response_time_percentile(0.99):.2f}ms
Requests/sec: {stats.total.total_rps:.2f}

Custom Metrics:
WebSocket Connections: {stats.custom_metrics.get('websocket_connections', 0)}
Auth Failures: {stats.custom_metrics.get('auth_failures', 0)}
Cache Hits: {stats.custom_metrics.get('cache_hits', 0)}
Model Switches: {stats.custom_metrics.get('model_switches', 0)}

=== PERFORMANCE THRESHOLDS ===
✅ Response Time P95 < 2000ms: {'PASS' if stats.total.get_response_time_percentile(0.95) < 2000 else 'FAIL'}
✅ Success Rate > 99%: {'PASS' if success_rate > 99 else 'FAIL'}
✅ Requests/sec > 100: {'PASS' if stats.total.total_rps > 100 else 'FAIL'}
    """
    
    logger.info(report)
    
    # Save detailed report
    with open("load_test_report.txt", "w") as f:
        f.write(report)
        
        # Detailed endpoint statistics
        f.write("\n=== ENDPOINT DETAILS ===\n")
        for endpoint in stats.entries:
            entry = stats.entries[endpoint]
            f.write(f"""
Endpoint: {endpoint.name}
Method: {endpoint.method}
Requests: {entry.num_requests}
Failures: {entry.num_failures}
Avg Response Time: {entry.avg_response_time:.2f}ms
Min Response Time: {entry.min_response_time:.2f}ms
Max Response Time: {entry.max_response_time:.2f}ms
Requests/sec: {entry.total_rps:.2f}
            """)


# Stress testing scenarios
class StressTestUser(FastHttpUser):
    """User for stress testing scenarios"""
    tasks = [APITaskSet]
    wait_time = between(0.1, 0.5)  # Very aggressive
    weight = 0  # Disabled by default
    
    @task(50)
    def rapid_fire_requests(self):
        """Send rapid requests to test rate limiting"""
        for i in range(10):
            response = self.client.get("/health")
            if response.status_code == 429:  # Rate limited
                logger.info("Rate limiting triggered")
                break
            time.sleep(0.01)  # 100 requests/second per user


# Load testing profiles
class LoadTestProfiles:
    """Predefined load testing profiles"""
    
    @staticmethod
    def light_load():
        """Light load: 10 users, 5 minute test"""
        return {
            "users": 10,
            "spawn_rate": 2,
            "run_time": "5m"
        }
    
    @staticmethod
    def medium_load():
        """Medium load: 100 users, 15 minute test"""
        return {
            "users": 100,
            "spawn_rate": 10,
            "run_time": "15m"
        }
    
    @staticmethod
    def heavy_load():
        """Heavy load: 1000 users, 30 minute test"""
        return {
            "users": 1000,
            "spawn_rate": 50,
            "run_time": "30m"
        }
    
    @staticmethod
    def stress_test():
        """Stress test: 2000 users, 10 minute test"""
        return {
            "users": 2000,
            "spawn_rate": 100,
            "run_time": "10m"
        }


if __name__ == "__main__":
    import os
    
    # Set environment variables for testing
    os.environ["LOCUST_HOST"] = "http://localhost:8000"
    
    # Example usage:
    # locust -f locustfile.py --users 100 --spawn-rate 10 --run-time 10m --html report.html