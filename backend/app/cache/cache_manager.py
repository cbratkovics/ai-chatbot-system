"""
Cache Manager - Centralized cache management with Redis backend
"""

import json
import logging
from typing import Any, Optional, Dict
import redis.asyncio as redis
from datetime import timedelta
import hashlib

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching operations with Redis backend"""
    
    def __init__(self, redis_url: str = None, default_ttl: int = 3600):
        """
        Initialize cache manager
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
        """
        # TODO: Get redis URL from settings when available
        self.redis_url = redis_url or "redis://localhost:6379"
        self.default_ttl = default_ttl
        self.redis_client = None
        self._connected = False
        
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url, 
                decode_responses=True,
                max_connections=10
            )
            await self.redis_client.ping()
            self._connected = True
            logger.info("Cache manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            self._connected = False
            
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self._connected:
            return None
            
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value) if self._is_json(value) else value
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
            
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if not provided)
            
        Returns:
            Success status
        """
        if not self._connected:
            return False
            
        try:
            ttl = ttl or self.default_ttl
            
            # Serialize value if needed
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            await self.redis_client.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
            
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache
        
        Args:
            key: Cache key
            
        Returns:
            Success status
        """
        if not self._connected:
            return False
            
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
            
    async def clear(self, pattern: str = "*") -> int:
        """
        Clear cache keys matching pattern
        
        Args:
            pattern: Key pattern (default: all keys)
            
        Returns:
            Number of keys deleted
        """
        if not self._connected:
            return 0
            
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                return await self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0
            
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache
        
        Args:
            key: Cache key
            
        Returns:
            True if exists
        """
        if not self._connected:
            return False
            
        try:
            return bool(await self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
            
    async def get_ttl(self, key: str) -> int:
        """
        Get remaining TTL for key
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no expiry, -2 if not exists
        """
        if not self._connected:
            return -2
            
        try:
            return await self.redis_client.ttl(key)
        except Exception as e:
            logger.error(f"Cache TTL error: {e}")
            return -2
            
    async def batch_get(self, keys: list) -> Dict[str, Any]:
        """
        Get multiple values from cache
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary of key-value pairs
        """
        if not self._connected:
            return {}
            
        try:
            values = await self.redis_client.mget(keys)
            result = {}
            for key, value in zip(keys, values):
                if value:
                    result[key] = json.loads(value) if self._is_json(value) else value
            return result
        except Exception as e:
            logger.error(f"Cache batch get error: {e}")
            return {}
            
    async def batch_set(self, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set multiple values in cache
        
        Args:
            items: Dictionary of key-value pairs
            ttl: TTL in seconds
            
        Returns:
            Success status
        """
        if not self._connected:
            return False
            
        try:
            ttl = ttl or self.default_ttl
            pipe = self.redis_client.pipeline()
            
            for key, value in items.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                pipe.setex(key, ttl, value)
                
            await pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Cache batch set error: {e}")
            return False
            
    async def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment counter in cache
        
        Args:
            key: Cache key
            amount: Increment amount
            
        Returns:
            New value
        """
        if not self._connected:
            return 0
            
        try:
            return await self.redis_client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Cache increment error: {e}")
            return 0
            
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Cache statistics dictionary
        """
        if not self._connected:
            return {"connected": False}
            
        try:
            info = await self.redis_client.info()
            return {
                "connected": True,
                "used_memory": info.get("used_memory_human", "0"),
                "connected_clients": info.get("connected_clients", 0),
                "total_keys": await self.redis_client.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"connected": False, "error": str(e)}
            
    @staticmethod
    def generate_key(*args) -> str:
        """
        Generate cache key from arguments
        
        Args:
            *args: Key components
            
        Returns:
            Cache key
        """
        key_str = ":".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()
        
    @staticmethod
    def _is_json(value: str) -> bool:
        """Check if string is JSON"""
        try:
            json.loads(value)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
            
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            self._connected = False