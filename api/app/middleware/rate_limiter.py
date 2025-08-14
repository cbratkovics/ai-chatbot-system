from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import redis.asyncio as redis
import time
from app.config import settings

class RateLimiter:
    def __init__(self):
        self.redis = redis.from_url(settings.redis_url, decode_responses=True)
        self.requests_limit = settings.rate_limit_requests
        self.window_seconds = settings.rate_limit_window
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """Token bucket algorithm implementation"""
        now = time.time()
        key = f"rate_limit:{client_id}"
        
        # Get current bucket state
        bucket_data = await self.redis.hgetall(key)
        
        if not bucket_data:
            # Initialize bucket
            await self.redis.hset(
                key,
                mapping={
                    "tokens": str(self.requests_limit - 1),
                    "last_refill": str(now)
                }
            )
            await self.redis.expire(key, self.window_seconds * 2)
            return True
        
        tokens = float(bucket_data.get("tokens", 0))
        last_refill = float(bucket_data.get("last_refill", now))
        
        # Calculate tokens to add
        time_passed = now - last_refill
        tokens_to_add = time_passed * (self.requests_limit / self.window_seconds)
        tokens = min(self.requests_limit, tokens + tokens_to_add)
        
        if tokens < 1:
            return False
        
        # Consume a token
        await self.redis.hset(
            key,
            mapping={
                "tokens": str(tokens - 1),
                "last_refill": str(now)
            }
        )
        await self.redis.expire(key, self.window_seconds * 2)
        
        return True

rate_limiter = RateLimiter()

async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    # Extract client ID (IP address or API key)
    client_id = request.client.host if request.client else "unknown"
    
    # Skip rate limiting for health checks
    if request.url.path == "/api/v1/health":
        return await call_next(request)
    
    # Check rate limit
    allowed = await rate_limiter.check_rate_limit(client_id)
    
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": f"Maximum {settings.rate_limit_requests} requests per {settings.rate_limit_window} seconds"
            }
        )
    
    response = await call_next(request)
    return response