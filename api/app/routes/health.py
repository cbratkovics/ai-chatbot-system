from fastapi import APIRouter
from datetime import datetime
import redis.asyncio as redis
from app.config import settings

router = APIRouter(prefix="/api/v1", tags=["health"])

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.api_version,
        "checks": {}
    }
    
    # Check Redis connection
    try:
        r = redis.from_url(settings.redis_url)
        await r.ping()
        await r.close()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
    
    return health_status