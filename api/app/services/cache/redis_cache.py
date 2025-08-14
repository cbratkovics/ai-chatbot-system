import json
from typing import Optional, Any
import redis.asyncio as redis
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self, prefix: str = "cache", ttl: int = 3600):
        self.redis = redis.from_url(settings.redis_url, decode_responses=True)
        self.prefix = prefix
        self.ttl = ttl
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        full_key = f"{self.prefix}:{key}"
        value = await self.redis.get(full_key)
        
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        full_key = f"{self.prefix}:{key}"
        
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        
        await self.redis.set(full_key, value, ex=ttl or self.ttl)
        logger.debug(f"Cached {full_key}")
    
    async def delete(self, key: str):
        """Delete value from cache"""
        full_key = f"{self.prefix}:{key}"
        await self.redis.delete(full_key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        full_key = f"{self.prefix}:{key}"
        return await self.redis.exists(full_key) > 0
    
    async def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern"""
        full_pattern = f"{self.prefix}:{pattern}"
        cursor = 0
        
        while True:
            cursor, keys = await self.redis.scan(cursor, match=full_pattern, count=100)
            
            if keys:
                await self.redis.delete(*keys)
            
            if cursor == 0:
                break