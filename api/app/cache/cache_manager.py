"""Multi-tier cache orchestration for the AI Chat Platform."""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class CacheTier(Enum):
    """Cache tier levels."""
    L1_MEMORY = "memory"  # In-memory cache (fastest)
    L2_REDIS = "redis"    # Redis cache (fast)
    L3_SEMANTIC = "semantic"  # Semantic cache (intelligent)


class MultiTierCache:
    """Multi-tier cache with automatic promotion/demotion."""
    
    def __init__(
        self,
        redis_client=None,
        semantic_cache=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize multi-tier cache.
        
        Args:
            redis_client: Redis client
            semantic_cache: Semantic cache instance
            config: Cache configuration
        """
        self.redis = redis_client
        self.semantic_cache = semantic_cache
        self.config = config or {}
        
        # L1 Memory cache
        self.memory_cache = {}
        self.memory_cache_size = self.config.get("memory_cache_size", 100)
        self.memory_cache_ttl = self.config.get("memory_cache_ttl", 300)  # 5 minutes
        
        # L2 Redis cache settings
        self.redis_ttl = self.config.get("redis_ttl", 3600)  # 1 hour
        
        # Access tracking for promotion
        self.access_counts = {}
        self.promotion_threshold = self.config.get("promotion_threshold", 3)
        
        # Statistics
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "misses": 0,
            "promotions": 0,
            "evictions": 0
        }
    
    async def get(
        self,
        key: str,
        tenant_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get from cache with tier traversal.
        
        Args:
            key: Cache key
            tenant_id: Tenant identifier
            
        Returns:
            Cached value if found
        """
        # Try L1 Memory
        memory_result = self._get_from_memory(key)
        if memory_result:
            self.stats["l1_hits"] += 1
            await self._track_access(key)
            return memory_result
        
        # Try L2 Redis
        if self.redis:
            redis_result = await self._get_from_redis(key)
            if redis_result:
                self.stats["l2_hits"] += 1
                await self._track_access(key)
                
                # Promote to L1 if accessed frequently
                if await self._should_promote(key):
                    self._store_in_memory(key, redis_result)
                    self.stats["promotions"] += 1
                
                return redis_result
        
        # Try L3 Semantic (if key represents a query)
        if self.semantic_cache:
            semantic_result = await self.semantic_cache.get(key, tenant_id)
            if semantic_result:
                self.stats["l3_hits"] += 1
                await self._track_access(key)
                
                # Promote to faster tiers
                if await self._should_promote(key):
                    await self._store_in_redis(key, semantic_result)
                    self._store_in_memory(key, semantic_result)
                    self.stats["promotions"] += 1
                
                return semantic_result
        
        self.stats["misses"] += 1
        return None
    
    async def set(
        self,
        key: str,
        value: Dict[str, Any],
        tier: CacheTier = CacheTier.L2_REDIS,
        tenant_id: Optional[str] = None,
        ttl: Optional[int] = None
    ):
        """Store in cache at specified tier.
        
        Args:
            key: Cache key
            value: Value to cache
            tier: Target cache tier
            tenant_id: Tenant identifier
            ttl: Time to live in seconds
        """
        if tier == CacheTier.L1_MEMORY:
            self._store_in_memory(key, value, ttl or self.memory_cache_ttl)
        
        elif tier == CacheTier.L2_REDIS and self.redis:
            await self._store_in_redis(key, value, ttl or self.redis_ttl)
        
        elif tier == CacheTier.L3_SEMANTIC and self.semantic_cache:
            # For semantic cache, key should be the query
            await self.semantic_cache.set(
                query=key,
                response=value.get("response", ""),
                tenant_id=tenant_id,
                model=value.get("model"),
                cost=value.get("cost", 0.0),
                metadata=value.get("metadata")
            )
    
    async def invalidate(
        self,
        pattern: Optional[str] = None,
        tenant_id: Optional[str] = None
    ):
        """Invalidate cache entries.
        
        Args:
            pattern: Pattern to match
            tenant_id: Tenant to clear
        """
        # Clear L1 Memory
        if pattern:
            keys_to_remove = [
                k for k in self.memory_cache.keys()
                if pattern in k
            ]
            for key in keys_to_remove:
                del self.memory_cache[key]
        elif tenant_id:
            keys_to_remove = [
                k for k in self.memory_cache.keys()
                if f":{tenant_id}:" in k
            ]
            for key in keys_to_remove:
                del self.memory_cache[key]
        
        # Clear L2 Redis
        if self.redis:
            if pattern:
                keys = await self.redis.keys(pattern)
                if keys:
                    await self.redis.delete(*keys)
            elif tenant_id:
                pattern = f"*:{tenant_id}:*"
                keys = await self.redis.keys(pattern)
                if keys:
                    await self.redis.delete(*keys)
        
        # Clear L3 Semantic
        if self.semantic_cache:
            await self.semantic_cache.invalidate(pattern, tenant_id)
    
    def _get_from_memory(self, key: str) -> Optional[Dict[str, Any]]:
        """Get from L1 memory cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found and not expired
        """
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() < entry["expires_at"]:
                return entry["value"]
            else:
                # Expired, remove it
                del self.memory_cache[key]
        return None
    
    async def _get_from_redis(self, key: str) -> Optional[Dict[str, Any]]:
        """Get from L2 Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found
        """
        try:
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
        return None
    
    def _store_in_memory(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: int = None
    ):
        """Store in L1 memory cache.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Time to live
        """
        # Evict if cache is full
        if len(self.memory_cache) >= self.memory_cache_size:
            self._evict_lru()
        
        self.memory_cache[key] = {
            "value": value,
            "expires_at": time.time() + (ttl or self.memory_cache_ttl),
            "last_accessed": time.time()
        }
    
    async def _store_in_redis(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: int = None
    ):
        """Store in L2 Redis cache.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Time to live
        """
        try:
            await self.redis.set(
                key,
                json.dumps(value),
                ex=ttl or self.redis_ttl
            )
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    async def _track_access(self, key: str):
        """Track access for promotion decisions.
        
        Args:
            key: Cache key
        """
        if key not in self.access_counts:
            self.access_counts[key] = {
                "count": 0,
                "first_access": time.time(),
                "last_access": time.time()
            }
        
        self.access_counts[key]["count"] += 1
        self.access_counts[key]["last_access"] = time.time()
    
    async def _should_promote(self, key: str) -> bool:
        """Determine if entry should be promoted to faster tier.
        
        Args:
            key: Cache key
            
        Returns:
            True if should promote
        """
        if key not in self.access_counts:
            return False
        
        access_info = self.access_counts[key]
        
        # Promote if accessed frequently
        if access_info["count"] >= self.promotion_threshold:
            # Reset count after promotion
            access_info["count"] = 0
            return True
        
        # Promote if accessed multiple times in short period
        time_window = 60  # 1 minute
        if (access_info["count"] >= 2 and 
            time.time() - access_info["first_access"] < time_window):
            access_info["count"] = 0
            return True
        
        return False
    
    def _evict_lru(self):
        """Evict least recently used entry from memory cache."""
        if not self.memory_cache:
            return
        
        # Find LRU entry
        lru_key = min(
            self.memory_cache.keys(),
            key=lambda k: self.memory_cache[k].get("last_accessed", 0)
        )
        
        del self.memory_cache[lru_key]
        self.stats["evictions"] += 1
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        total_requests = sum([
            self.stats["l1_hits"],
            self.stats["l2_hits"],
            self.stats["l3_hits"],
            self.stats["misses"]
        ])
        
        hit_rate = 0.0
        if total_requests > 0:
            total_hits = (
                self.stats["l1_hits"] +
                self.stats["l2_hits"] +
                self.stats["l3_hits"]
            )
            hit_rate = total_hits / total_requests
        
        return {
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "l1_hits": self.stats["l1_hits"],
            "l2_hits": self.stats["l2_hits"],
            "l3_hits": self.stats["l3_hits"],
            "misses": self.stats["misses"],
            "promotions": self.stats["promotions"],
            "evictions": self.stats["evictions"],
            "memory_cache_size": len(self.memory_cache),
            "memory_cache_max": self.memory_cache_size
        }


class ResponseCache:
    """Specialized cache for LLM responses."""
    
    def __init__(
        self,
        multi_tier_cache: MultiTierCache,
        cost_calculator=None
    ):
        """Initialize response cache.
        
        Args:
            multi_tier_cache: Multi-tier cache instance
            cost_calculator: Cost calculation utility
        """
        self.cache = multi_tier_cache
        self.cost_calculator = cost_calculator
    
    async def get_response(
        self,
        query: str,
        model: str,
        tenant_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Optional[Dict[str, Any]]:
        """Get cached response.
        
        Args:
            query: User query
            model: Model name
            tenant_id: Tenant identifier
            temperature: Temperature setting
            max_tokens: Max tokens setting
            
        Returns:
            Cached response if found
        """
        # Create cache key with parameters
        cache_key = self._create_response_key(
            query,
            model,
            tenant_id,
            temperature,
            max_tokens
        )
        
        result = await self.cache.get(cache_key, tenant_id)
        
        if result:
            # Add cache metadata
            result["from_cache"] = True
            result["cache_key"] = cache_key
            
            # Calculate cost saved
            if self.cost_calculator:
                cost_saved = self.cost_calculator.calculate(
                    model,
                    result.get("token_count", 0)
                )
                result["cost_saved"] = cost_saved
        
        return result
    
    async def store_response(
        self,
        query: str,
        response: str,
        model: str,
        tenant_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        token_count: int = 0,
        metadata: Optional[Dict] = None
    ):
        """Store response in cache.
        
        Args:
            query: User query
            response: LLM response
            model: Model used
            tenant_id: Tenant identifier
            temperature: Temperature setting
            max_tokens: Max tokens setting
            token_count: Tokens used
            metadata: Additional metadata
        """
        cache_key = self._create_response_key(
            query,
            model,
            tenant_id,
            temperature,
            max_tokens
        )
        
        cache_value = {
            "query": query,
            "response": response,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "token_count": token_count,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Determine cache tier based on model and token count
        if token_count < 500:
            tier = CacheTier.L1_MEMORY
        elif token_count < 2000:
            tier = CacheTier.L2_REDIS
        else:
            tier = CacheTier.L3_SEMANTIC
        
        await self.cache.set(
            cache_key,
            cache_value,
            tier,
            tenant_id
        )
    
    def _create_response_key(
        self,
        query: str,
        model: str,
        tenant_id: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Create cache key for response.
        
        Args:
            query: Query text
            model: Model name
            tenant_id: Tenant ID
            temperature: Temperature
            max_tokens: Max tokens
            
        Returns:
            Cache key
        """
        # Include parameters that affect response
        key_data = {
            "query": query,
            "model": model,
            "tenant": tenant_id or "global",
            "temp": round(temperature, 2),
            "max_tokens": max_tokens
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()
        
        return f"response:{tenant_id or 'global'}:{model}:{key_hash}"


class CacheWarmer:
    """Proactively warm cache with common queries."""
    
    def __init__(
        self,
        response_cache: ResponseCache,
        llm_service=None
    ):
        """Initialize cache warmer.
        
        Args:
            response_cache: Response cache instance
            llm_service: LLM service for generating responses
        """
        self.cache = response_cache
        self.llm_service = llm_service
    
    async def warm_cache(
        self,
        queries: List[str],
        models: List[str],
        tenant_id: Optional[str] = None
    ):
        """Warm cache with pre-generated responses.
        
        Args:
            queries: List of queries to cache
            models: List of models to use
            tenant_id: Tenant identifier
        """
        tasks = []
        
        for query in queries:
            for model in models:
                task = self._generate_and_cache(
                    query,
                    model,
                    tenant_id
                )
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Cache warmed with {success_count}/{len(tasks)} entries")
    
    async def _generate_and_cache(
        self,
        query: str,
        model: str,
        tenant_id: Optional[str]
    ):
        """Generate and cache single response.
        
        Args:
            query: Query text
            model: Model to use
            tenant_id: Tenant identifier
        """
        try:
            if not self.llm_service:
                return
            
            # Check if already cached
            existing = await self.cache.get_response(
                query,
                model,
                tenant_id
            )
            
            if existing:
                return  # Already cached
            
            # Generate response
            response = await self.llm_service.generate(
                query,
                model=model,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Store in cache
            await self.cache.store_response(
                query=query,
                response=response["text"],
                model=model,
                tenant_id=tenant_id,
                token_count=response.get("token_count", 0)
            )
            
        except Exception as e:
            logger.error(f"Failed to warm cache for query: {e}")
    
    async def analyze_patterns(
        self,
        tenant_id: Optional[str] = None,
        days: int = 7
    ) -> List[str]:
        """Analyze query patterns to identify cache candidates.
        
        Args:
            tenant_id: Tenant identifier
            days: Days to analyze
            
        Returns:
            List of common queries to cache
        """
        # This would analyze historical queries to find patterns
        # For now, return common examples
        return [
            "What is the weather today?",
            "Tell me a joke",
            "What can you help me with?",
            "How do I get started?",
            "What are your capabilities?"
        ]