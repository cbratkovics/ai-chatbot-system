"""Advanced semantic cache with vector similarity search using Pinecone/FAISS."""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    query: str
    response: str
    embedding: List[float]
    model: str
    timestamp: float
    hit_count: int = 0
    tenant_id: Optional[str] = None
    metadata: Dict[str, Any] = None


class SemanticSearchEngine:
    """Semantic search engine with multiple backend support."""
    
    def __init__(
        self,
        backend: str = "faiss",
        dimension: int = 1536,
        similarity_threshold: float = 0.85
    ):
        """Initialize semantic search engine.
        
        Args:
            backend: Backend to use (faiss, pinecone)
            dimension: Embedding dimension
            similarity_threshold: Minimum similarity for cache hit
        """
        self.backend = backend
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold
        self.index = None
        self.metadata_store = {}
        
        if backend == "faiss":
            self._init_faiss()
        elif backend == "pinecone":
            self._init_pinecone()
    
    def _init_faiss(self):
        """Initialize FAISS index."""
        try:
            import faiss
            # Use HNSW index for better performance
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 100
            logger.info("Initialized FAISS HNSW index")
        except ImportError:
            logger.warning("FAISS not available, using numpy fallback")
            self.index = None
    
    def _init_pinecone(self):
        """Initialize Pinecone index."""
        try:
            import pinecone
            import os
            
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENV", "us-west1-gcp")
            )
            
            index_name = os.getenv("PINECONE_INDEX", "chatbot-cache")
            
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    pod_type="p1.x1"
                )
            
            self.index = pinecone.Index(index_name)
            logger.info(f"Connected to Pinecone index: {index_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            self.index = None
    
    async def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        tenant_id: Optional[str] = None
    ) -> List[Tuple[str, float, Dict]]:
        """Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            tenant_id: Filter by tenant
            
        Returns:
            List of (id, similarity, metadata) tuples
        """
        if self.backend == "faiss":
            return await self._search_faiss(query_embedding, k, tenant_id)
        elif self.backend == "pinecone":
            return await self._search_pinecone(query_embedding, k, tenant_id)
        else:
            return await self._search_numpy(query_embedding, k, tenant_id)
    
    async def _search_faiss(
        self,
        query_embedding: List[float],
        k: int,
        tenant_id: Optional[str]
    ) -> List[Tuple[str, float, Dict]]:
        """Search using FAISS."""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_vec = np.array(query_embedding).astype('float32').reshape(1, -1)
        
        # Search for more results to filter by tenant
        search_k = min(k * 10, self.index.ntotal)
        distances, indices = self.index.search(query_vec, search_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            metadata = self.metadata_store.get(str(idx), {})
            
            # Filter by tenant if specified
            if tenant_id and metadata.get("tenant_id") != tenant_id:
                continue
            
            # Convert L2 distance to cosine similarity
            similarity = 1 - (dist / 2)
            
            if similarity >= self.similarity_threshold:
                results.append((str(idx), float(similarity), metadata))
            
            if len(results) >= k:
                break
        
        return results
    
    async def _search_pinecone(
        self,
        query_embedding: List[float],
        k: int,
        tenant_id: Optional[str]
    ) -> List[Tuple[str, float, Dict]]:
        """Search using Pinecone."""
        if self.index is None:
            return []
        
        filter_dict = {"tenant_id": tenant_id} if tenant_id else None
        
        try:
            response = self.index.query(
                vector=query_embedding,
                top_k=k,
                filter=filter_dict,
                include_metadata=True
            )
            
            results = []
            for match in response.matches:
                if match.score >= self.similarity_threshold:
                    results.append((
                        match.id,
                        match.score,
                        match.metadata or {}
                    ))
            
            return results
        except Exception as e:
            logger.error(f"Pinecone search error: {e}")
            return []
    
    async def _search_numpy(
        self,
        query_embedding: List[float],
        k: int,
        tenant_id: Optional[str]
    ) -> List[Tuple[str, float, Dict]]:
        """Fallback numpy search."""
        if not self.metadata_store:
            return []
        
        query_vec = np.array(query_embedding)
        results = []
        
        for key, metadata in self.metadata_store.items():
            if tenant_id and metadata.get("tenant_id") != tenant_id:
                continue
            
            if "embedding" not in metadata:
                continue
            
            stored_vec = np.array(metadata["embedding"])
            
            # Cosine similarity
            similarity = np.dot(query_vec, stored_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(stored_vec)
            )
            
            if similarity >= self.similarity_threshold:
                results.append((key, float(similarity), metadata))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    async def add(
        self,
        id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ):
        """Add vector to index.
        
        Args:
            id: Unique identifier
            embedding: Vector embedding
            metadata: Associated metadata
        """
        if self.backend == "faiss":
            await self._add_faiss(id, embedding, metadata)
        elif self.backend == "pinecone":
            await self._add_pinecone(id, embedding, metadata)
        else:
            await self._add_numpy(id, embedding, metadata)
    
    async def _add_faiss(
        self,
        id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ):
        """Add to FAISS index."""
        if self.index is None:
            import faiss
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        
        vec = np.array(embedding).astype('float32').reshape(1, -1)
        idx = self.index.ntotal
        self.index.add(vec)
        
        # Store metadata
        metadata["embedding"] = embedding
        self.metadata_store[str(idx)] = metadata
    
    async def _add_pinecone(
        self,
        id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ):
        """Add to Pinecone index."""
        if self.index is None:
            return
        
        try:
            self.index.upsert([(id, embedding, metadata)])
        except Exception as e:
            logger.error(f"Failed to add to Pinecone: {e}")
    
    async def _add_numpy(
        self,
        id: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ):
        """Add to numpy store."""
        metadata["embedding"] = embedding
        self.metadata_store[id] = metadata
    
    async def delete(self, id: str):
        """Delete vector from index.
        
        Args:
            id: Vector ID to delete
        """
        if self.backend == "pinecone" and self.index:
            try:
                self.index.delete([id])
            except Exception as e:
                logger.error(f"Failed to delete from Pinecone: {e}")
        
        # Remove from metadata store
        self.metadata_store.pop(id, None)
    
    async def clear_tenant(self, tenant_id: str):
        """Clear all vectors for a tenant.
        
        Args:
            tenant_id: Tenant identifier
        """
        if self.backend == "pinecone" and self.index:
            try:
                self.index.delete(filter={"tenant_id": tenant_id})
            except Exception as e:
                logger.error(f"Failed to clear tenant data: {e}")
        
        # Clear from metadata store
        keys_to_delete = [
            k for k, v in self.metadata_store.items()
            if v.get("tenant_id") == tenant_id
        ]
        for key in keys_to_delete:
            del self.metadata_store[key]


class SemanticCache:
    """High-performance semantic cache with cost optimization."""
    
    def __init__(
        self,
        search_engine: SemanticSearchEngine,
        redis_client=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize semantic cache.
        
        Args:
            search_engine: Semantic search backend
            redis_client: Redis client for metadata
            config: Cache configuration
        """
        self.search_engine = search_engine
        self.redis = redis_client
        self.config = config or {}
        
        self.ttl_exact = self.config.get("ttl_exact", 3600)  # 1 hour
        self.ttl_semantic = self.config.get("ttl_semantic", 1800)  # 30 minutes
        self.max_cache_size = self.config.get("max_cache_size", 10000)
        self.embedding_model = self.config.get("embedding_model", "text-embedding-ada-002")
        
        self.stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "cost_saved": 0.0
        }
    
    async def get(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        model: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached response for query.
        
        Args:
            query: Query text
            tenant_id: Tenant identifier
            model: Model preference
            
        Returns:
            Cached response if found
        """
        # Try exact match first
        exact_result = await self._get_exact(query, tenant_id, model)
        if exact_result:
            self.stats["exact_hits"] += 1
            await self._track_hit(tenant_id, "exact", exact_result.get("cost_saved", 0))
            return exact_result
        
        # Try semantic match
        semantic_result = await self._get_semantic(query, tenant_id, model)
        if semantic_result:
            self.stats["semantic_hits"] += 1
            await self._track_hit(tenant_id, "semantic", semantic_result.get("cost_saved", 0))
            return semantic_result
        
        self.stats["misses"] += 1
        return None
    
    async def _get_exact(
        self,
        query: str,
        tenant_id: Optional[str],
        model: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Get exact match from cache.
        
        Args:
            query: Query text
            tenant_id: Tenant ID
            model: Model name
            
        Returns:
            Cached response if found
        """
        if not self.redis:
            return None
        
        # Create cache key
        cache_key = self._create_cache_key(query, tenant_id, model)
        
        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                
                # Check if not expired
                if time.time() - data.get("timestamp", 0) < self.ttl_exact:
                    return data
                else:
                    # Expired, remove it
                    await self.redis.delete(cache_key)
        except Exception as e:
            logger.error(f"Error getting exact cache: {e}")
        
        return None
    
    async def _get_semantic(
        self,
        query: str,
        tenant_id: Optional[str],
        model: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Get semantic match from cache.
        
        Args:
            query: Query text
            tenant_id: Tenant ID
            model: Model name
            
        Returns:
            Cached response if found
        """
        try:
            # Generate embedding for query
            query_embedding = await self._generate_embedding(query)
            
            # Search for similar queries
            results = await self.search_engine.search(
                query_embedding,
                k=1,
                tenant_id=tenant_id
            )
            
            if results:
                id, similarity, metadata = results[0]
                
                # Check similarity threshold
                if similarity >= self.search_engine.similarity_threshold:
                    # Check if not expired
                    if time.time() - metadata.get("timestamp", 0) < self.ttl_semantic:
                        # Check model compatibility
                        if not model or metadata.get("model") == model:
                            return {
                                "response": metadata.get("response"),
                                "model": metadata.get("model"),
                                "similarity": similarity,
                                "cached": True,
                                "cache_type": "semantic",
                                "original_query": metadata.get("query"),
                                "cost_saved": metadata.get("cost", 0)
                            }
        except Exception as e:
            logger.error(f"Error getting semantic cache: {e}")
        
        return None
    
    async def set(
        self,
        query: str,
        response: str,
        tenant_id: Optional[str] = None,
        model: Optional[str] = None,
        cost: float = 0.0,
        metadata: Optional[Dict] = None
    ):
        """Store response in cache.
        
        Args:
            query: Query text
            response: Response text
            tenant_id: Tenant identifier
            model: Model used
            cost: Cost of generating response
            metadata: Additional metadata
        """
        try:
            # Generate embedding
            query_embedding = await self._generate_embedding(query)
            
            # Create cache entry
            cache_data = {
                "query": query,
                "response": response,
                "model": model,
                "tenant_id": tenant_id,
                "timestamp": time.time(),
                "cost": cost,
                "metadata": metadata or {}
            }
            
            # Store exact match in Redis
            if self.redis:
                cache_key = self._create_cache_key(query, tenant_id, model)
                await self.redis.set(
                    cache_key,
                    json.dumps(cache_data),
                    ex=self.ttl_exact
                )
            
            # Store in vector index for semantic search
            entry_id = self._create_entry_id(query, tenant_id, model)
            await self.search_engine.add(
                entry_id,
                query_embedding,
                cache_data
            )
            
            # Manage cache size
            await self._evict_if_needed()
            
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
    
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
        if tenant_id:
            await self.search_engine.clear_tenant(tenant_id)
            
            if self.redis:
                # Clear Redis entries for tenant
                pattern = f"cache:{tenant_id}:*"
                keys = await self.redis.keys(pattern)
                if keys:
                    await self.redis.delete(*keys)
        
        elif pattern and self.redis:
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
    
    async def get_stats(
        self,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get cache statistics.
        
        Args:
            tenant_id: Filter by tenant
            
        Returns:
            Cache statistics
        """
        stats = {
            "exact_hits": self.stats["exact_hits"],
            "semantic_hits": self.stats["semantic_hits"],
            "misses": self.stats["misses"],
            "hit_rate": 0.0,
            "cost_saved": self.stats["cost_saved"]
        }
        
        total = stats["exact_hits"] + stats["semantic_hits"] + stats["misses"]
        if total > 0:
            stats["hit_rate"] = (stats["exact_hits"] + stats["semantic_hits"]) / total
        
        if tenant_id and self.redis:
            # Get tenant-specific stats
            tenant_key = f"cache_stats:{tenant_id}"
            tenant_stats = await self.redis.hgetall(tenant_key)
            if tenant_stats:
                stats["tenant_stats"] = {
                    k.decode(): float(v.decode())
                    for k, v in tenant_stats.items()
                }
        
        return stats
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI()
            
            response = await client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return random embedding as fallback
            return np.random.rand(self.search_engine.dimension).tolist()
    
    def _create_cache_key(
        self,
        query: str,
        tenant_id: Optional[str],
        model: Optional[str]
    ) -> str:
        """Create cache key.
        
        Args:
            query: Query text
            tenant_id: Tenant ID
            model: Model name
            
        Returns:
            Cache key
        """
        parts = ["cache"]
        if tenant_id:
            parts.append(tenant_id)
        if model:
            parts.append(model)
        
        query_hash = hashlib.md5(query.encode()).hexdigest()
        parts.append(query_hash)
        
        return ":".join(parts)
    
    def _create_entry_id(
        self,
        query: str,
        tenant_id: Optional[str],
        model: Optional[str]
    ) -> str:
        """Create unique entry ID.
        
        Args:
            query: Query text
            tenant_id: Tenant ID
            model: Model name
            
        Returns:
            Entry ID
        """
        data = f"{tenant_id or 'global'}:{model or 'any'}:{query}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    async def _track_hit(
        self,
        tenant_id: Optional[str],
        hit_type: str,
        cost_saved: float
    ):
        """Track cache hit metrics.
        
        Args:
            tenant_id: Tenant ID
            hit_type: Type of hit
            cost_saved: Cost saved
        """
        self.stats["cost_saved"] += cost_saved
        
        if tenant_id and self.redis:
            try:
                tenant_key = f"cache_stats:{tenant_id}"
                await self.redis.hincrby(tenant_key, f"{hit_type}_hits", 1)
                await self.redis.hincrbyfloat(tenant_key, "cost_saved", cost_saved)
                await self.redis.expire(tenant_key, 86400)  # 24 hours
            except Exception as e:
                logger.error(f"Error tracking cache hit: {e}")
    
    async def _evict_if_needed(self):
        """Evict old entries if cache is full."""
        # This would implement LRU or other eviction strategies
        # For now, we'll rely on TTL-based eviction
        pass


class CacheManager:
    """Manages multiple cache layers and strategies."""
    
    def __init__(self, redis_client=None):
        """Initialize cache manager.
        
        Args:
            redis_client: Redis client
        """
        self.redis = redis_client
        
        # Initialize search engine
        backend = "faiss"  # or "pinecone" based on config
        self.search_engine = SemanticSearchEngine(
            backend=backend,
            dimension=1536,
            similarity_threshold=0.85
        )
        
        # Initialize semantic cache
        self.semantic_cache = SemanticCache(
            search_engine=self.search_engine,
            redis_client=redis_client,
            config={
                "ttl_exact": 3600,
                "ttl_semantic": 1800,
                "max_cache_size": 10000
            }
        )
    
    async def get(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        model: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get from cache.
        
        Args:
            query: Query text
            tenant_id: Tenant ID
            model: Model preference
            
        Returns:
            Cached response if found
        """
        return await self.semantic_cache.get(query, tenant_id, model)
    
    async def set(
        self,
        query: str,
        response: str,
        tenant_id: Optional[str] = None,
        model: Optional[str] = None,
        cost: float = 0.0,
        metadata: Optional[Dict] = None
    ):
        """Store in cache.
        
        Args:
            query: Query text
            response: Response text
            tenant_id: Tenant ID
            model: Model used
            cost: Generation cost
            metadata: Additional data
        """
        await self.semantic_cache.set(
            query,
            response,
            tenant_id,
            model,
            cost,
            metadata
        )
    
    async def get_cost_savings(
        self,
        tenant_id: Optional[str] = None,
        period: str = "day"
    ) -> Dict[str, float]:
        """Calculate cost savings from cache.
        
        Args:
            tenant_id: Tenant ID
            period: Time period
            
        Returns:
            Cost savings data
        """
        stats = await self.semantic_cache.get_stats(tenant_id)
        
        return {
            "total_saved": stats["cost_saved"],
            "hit_rate": stats["hit_rate"],
            "reduction_percentage": stats["hit_rate"] * 100
        }