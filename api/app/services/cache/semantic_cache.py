import numpy as np
from typing import Optional, List, Tuple
import json
import hashlib
from sentence_transformers import SentenceTransformer
import redis.asyncio as redis
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.85):
        self.redis = redis.from_url(settings.redis_url, decode_responses=True)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = similarity_threshold
        self.cache_ttl = 3600  # 1 hour
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return f"cache:{hashlib.md5(text.encode()).hexdigest()}"
    
    async def get_cached_response(
        self, 
        query: str, 
        context: Optional[List[str]] = None
    ) -> Optional[Tuple[str, float]]:
        """Retrieve cached response if similar query exists"""
        
        # Get query embedding
        query_embedding = self.encoder.encode(query).tolist()
        
        # Search for similar queries
        cache_pattern = "cache:*"
        cursor = 0
        best_match = None
        best_score = 0.0
        
        while True:
            cursor, keys = await self.redis.scan(
                cursor, 
                match=cache_pattern, 
                count=100
            )
            
            for key in keys:
                cached_data = await self.redis.hgetall(key)
                if not cached_data or "embedding" not in cached_data:
                    continue
                
                # Calculate similarity
                cached_embedding = json.loads(cached_data["embedding"])
                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                
                if similarity > self.similarity_threshold and similarity > best_score:
                    best_score = similarity
                    best_match = cached_data["response"]
            
            if cursor == 0:
                break
        
        if best_match:
            logger.info(f"Cache hit with similarity {best_score:.3f}")
            return best_match, best_score
        
        return None
    
    async def cache_response(
        self, 
        query: str, 
        response: str, 
        context: Optional[List[str]] = None
    ):
        """Cache response with semantic embedding"""
        
        # Generate embedding
        query_embedding = self.encoder.encode(query).tolist()
        
        # Store in Redis
        cache_key = self._generate_cache_key(query)
        await self.redis.hset(
            cache_key,
            mapping={
                "query": query,
                "response": response,
                "embedding": json.dumps(query_embedding),
                "context": json.dumps(context or [])
            }
        )
        
        # Set TTL
        await self.redis.expire(cache_key, self.cache_ttl)
        
        logger.info(f"Cached response for query: {query[:50]}...")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)