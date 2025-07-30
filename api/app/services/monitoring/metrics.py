from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response
import time
from functools import wraps
from typing import Callable

# Define metrics
request_count = Counter(
    'chatbot_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'chatbot_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

active_sessions = Gauge(
    'chatbot_active_sessions',
    'Number of active chat sessions'
)

tokens_used = Counter(
    'chatbot_tokens_total',
    'Total tokens used',
    ['model', 'provider']
)

api_cost = Counter(
    'chatbot_api_cost_dollars',
    'Total API cost in dollars',
    ['model', 'provider']
)

cache_hits = Counter(
    'chatbot_cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

cache_misses = Counter(
    'chatbot_cache_misses_total',
    'Total cache misses',
    ['cache_type']
)

model_latency = Histogram(
    'chatbot_model_latency_seconds',
    'Model response latency',
    ['model', 'provider'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

error_count = Counter(
    'chatbot_errors_total',
    'Total number of errors',
    ['error_type', 'endpoint']
)

# Metrics endpoint
async def metrics_endpoint():
    return Response(content=generate_latest(), media_type="text/plain")

# Decorator for tracking metrics
def track_metrics(endpoint: str):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                error_count.labels(
                    error_type=type(e).__name__,
                    endpoint=endpoint
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                request_count.labels(
                    method="POST",
                    endpoint=endpoint,
                    status=status
                ).inc()
                request_duration.labels(
                    method="POST",
                    endpoint=endpoint
                ).observe(duration)
        
        return wrapper
    return decorator