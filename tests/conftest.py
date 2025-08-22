"""Shared pytest fixtures for all test modules."""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from faker import Faker
from httpx import AsyncClient
from redis import asyncio as aioredis
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

fake = Faker()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.setex = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.exists = AsyncMock(return_value=0)
    redis_mock.expire = AsyncMock(return_value=True)
    redis_mock.ttl = AsyncMock(return_value=-1)
    redis_mock.hget = AsyncMock(return_value=None)
    redis_mock.hset = AsyncMock(return_value=1)
    redis_mock.zadd = AsyncMock(return_value=1)
    redis_mock.zrange = AsyncMock(return_value=[])
    return redis_mock


@pytest.fixture
def mock_database():
    """Mock database session for testing."""
    session = AsyncMock(spec=AsyncSession)
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.execute = AsyncMock()
    session.query = MagicMock()
    return session


@pytest.fixture
def sample_chat_request():
    """Sample chat request payload."""
    return {
        "message": "What is the weather like today?",
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 150,
        "user_id": fake.uuid4(),
        "tenant_id": fake.uuid4(),
        "session_id": fake.uuid4(),
        "metadata": {"source": "web", "timestamp": datetime.utcnow().isoformat()},
    }


@pytest.fixture
def sample_chat_response():
    """Sample chat response payload."""
    return {
        "id": fake.uuid4(),
        "object": "chat.completion",
        "created": int(datetime.utcnow().timestamp()),
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I don't have access to real-time weather data. Please check a weather service for current conditions.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
    }


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    client = AsyncMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    client = AsyncMock()
    client.messages.create = AsyncMock()
    return client


@pytest.fixture
def auth_headers():
    """Sample authentication headers."""
    return {"Authorization": f"Bearer {fake.sha256()}", "X-Tenant-ID": fake.uuid4(), "X-Request-ID": fake.uuid4()}


@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection for testing."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_text = AsyncMock(return_value='{"type": "message", "data": "test"}')
    ws.receive_json = AsyncMock(return_value={"type": "message", "data": "test"})
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def performance_metrics():
    """Performance metrics for validation."""
    return {
        "p95_latency_ms": 200,
        "concurrent_users": 100,
        "cache_hit_rate": 0.3,
        "requests_per_second": 1000,
        "error_rate": 0.01,
    }


@pytest.fixture
def tenant_config():
    """Sample tenant configuration."""
    return {
        "tenant_id": fake.uuid4(),
        "name": fake.company(),
        "tier": "enterprise",
        "rate_limits": {"requests_per_minute": 1000, "tokens_per_day": 1000000, "concurrent_connections": 50},
        "features": {
            "semantic_cache": True,
            "model_switching": True,
            "custom_models": ["gpt-4", "claude-3"],
            "data_retention_days": 90,
        },
        "created_at": datetime.utcnow().isoformat(),
        "status": "active",
    }


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for semantic cache testing."""
    return {
        "text": "What is the weather like today?",
        "embedding": [fake.random.random() for _ in range(1536)],
        "model": "text-embedding-ada-002",
        "metadata": {"timestamp": datetime.utcnow().isoformat(), "tokens": 8},
    }


@pytest.fixture
async def async_http_client():
    """Async HTTP client for API testing."""
    async with AsyncClient(base_url="http://test") as client:
        yield client


@pytest.fixture
def rate_limit_config():
    """Rate limiting configuration."""
    return {"window_seconds": 60, "max_requests": 100, "burst_size": 20, "strategy": "token_bucket"}


@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector for monitoring."""
    collector = MagicMock()
    collector.record_latency = MagicMock()
    collector.increment_counter = MagicMock()
    collector.record_gauge = MagicMock()
    collector.record_histogram = MagicMock()
    return collector


@pytest.fixture
def cache_config():
    """Cache configuration for testing."""
    return {
        "ttl_seconds": 3600,
        "max_size_mb": 100,
        "eviction_policy": "lru",
        "similarity_threshold": 0.85,
        "embedding_dimension": 1536,
    }


@pytest.fixture
def load_test_data():
    """Load test data from fixtures."""

    def _load(filename):
        filepath = os.path.join(os.path.dirname(__file__), "fixtures", filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
        return {}

    return _load


@pytest.fixture
def mock_stream_response():
    """Mock streaming response for SSE/WebSocket testing."""

    async def _stream():
        chunks = ["Hello", " from", " the", " AI", " assistant!"]
        for chunk in chunks:
            await asyncio.sleep(0.01)
            yield {
                "id": fake.uuid4(),
                "object": "chat.completion.chunk",
                "created": int(datetime.utcnow().timestamp()),
                "model": "gpt-4",
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
            }

    return _stream


@pytest.fixture
def environment_variables(monkeypatch):
    """Set test environment variables."""
    env_vars = {
        "ENVIRONMENT": "test",
        "DEBUG": "true",
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
        "REDIS_URL": "redis://localhost:6379/0",
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "JWT_SECRET": fake.sha256(),
        "ENCRYPTION_KEY": fake.sha256(),
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars
