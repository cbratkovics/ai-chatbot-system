import pytest
import numpy as np
from app.services.cache.semantic_cache import SemanticCache

@pytest.mark.asyncio
async def test_semantic_cache_similarity():
    cache = SemanticCache(similarity_threshold=0.8)
    
    # Test cosine similarity calculation
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    similarity = cache._cosine_similarity(vec1, vec2)
    assert similarity == 1.0
    
    # Test orthogonal vectors
    vec3 = [0.0, 1.0, 0.0]
    similarity = cache._cosine_similarity(vec1, vec3)
    assert similarity == 0.0

@pytest.mark.asyncio
async def test_cache_operations():
    cache = SemanticCache()
    
    # Test cache miss
    result = await cache.get_cached_response("test query")
    assert result is None
    
    # Test cache storage
    await cache.cache_response("test query", "test response")
    
    # Test exact match (would need similar query to test semantic matching)
    # In real tests, you'd want to test with actual similar queries