"""Global test configuration and fixtures."""
import asyncio
from collections.abc import AsyncGenerator

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from api.core.config import settings
from api.main import app


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
async def db_session():
    """Create database session for testing."""
    engine = create_async_engine(settings.DATABASE_URL_TEST)
    async_session = sessionmaker(engine, class_=AsyncSession)

    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
def auth_headers():
    """Generate authentication headers."""
    return {"Authorization": "Bearer test-token"}
