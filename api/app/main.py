from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from app.config import settings
from app.routes import chat, websocket, upload
from app.routes.health import router as health_router
from app.middleware.rate_limiter import rate_limit_middleware
from app.middleware.error_handler import ErrorHandlerMiddleware
from app.services.monitoring.metrics import metrics_endpoint

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
    
    # Initialize functions for function calling
    from app.services.functions.init_functions import initialize_functions
    initialize_functions()
    
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
app.add_middleware(ErrorHandlerMiddleware)
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
app.include_router(chat.router)
app.include_router(websocket.router)
app.include_router(upload.router)

# Add metrics endpoint
@app.get("/metrics")
async def get_metrics():
    return await metrics_endpoint()

@app.get("/")
async def root():
    return {
        "message": "AI Conversational Platform API",
        "version": settings.api_version,
        "docs": "/docs"
    }