from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from .config import settings
from .routes.health import router as health_router
from .tenancy.rate_limiter import rate_limit_middleware
from .monitoring.health_checks import HealthCheckManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting AI Conversational Platform")
    
    # Initialize health checks
    health_manager = HealthCheckManager()
    await health_manager.start_background_checks()
    
    yield
    # Shutdown
    logger.info("Shutting down AI Conversational Platform")

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    debug=settings.debug,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting
app.middleware("http")(rate_limit_middleware)

# Include routers
app.include_router(health_router)

@app.get("/")
async def root():
    return {
        "message": "AI Conversational Platform API",
        "version": settings.api_version,
        "docs": "/docs"
    }