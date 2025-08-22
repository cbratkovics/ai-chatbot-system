"""
AI Chatbot Demo - Streamlined FastAPI Application
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import sys

# Import demo configuration
from app.demo_config import demo_settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, demo_settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("🚀 Starting AI Chatbot Demo")
    
    # Validate configuration
    is_valid, error_msg = demo_settings.validate_config()
    if not is_valid:
        logger.error(f"❌ Configuration Error: {error_msg}")
        logger.error("📝 Please add your API keys to the .env file")
        raise ValueError(error_msg)
    
    # Log available providers
    providers = demo_settings.get_available_providers()
    logger.info(f"✅ Available LLM Providers: {', '.join(providers)}")
    logger.info(f"📊 Default Model: {demo_settings.get_default_model()}")
    
    # Initialize cache if Redis is available
    try:
        from app.cache.cache_manager import CacheManager
        cache = CacheManager(redis_url=demo_settings.redis_url)
        await cache.initialize()
        app.state.cache = cache
        logger.info("✅ Cache initialized")
    except Exception as e:
        logger.warning(f"⚠️ Cache unavailable (will continue without caching): {e}")
        app.state.cache = None
    
    logger.info("✅ Demo API ready at http://localhost:8000")
    logger.info("📚 API Documentation: http://localhost:8000/docs")
    
    yield
    
    # Cleanup
    logger.info("👋 Shutting down AI Chatbot Demo")
    if hasattr(app.state, 'cache') and app.state.cache:
        await app.state.cache.close()

# Create FastAPI app
app = FastAPI(
    title="🤖 AI Chatbot Demo",
    description="""
    A production-ready AI chatbot with multiple LLM providers.
    
    ## Features
    - 🔄 Multiple LLM providers (OpenAI, Anthropic)
    - 💾 Intelligent caching
    - 🔌 WebSocket support
    - 📊 Real-time streaming
    - 🔒 Rate limiting
    
    ## Quick Start
    1. Add your API keys to `.env` file
    2. Choose a model from the available providers
    3. Start chatting!
    """,
    version="1.0.0-demo",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from app.routes.health import router as health_router
from app.routes.chat import router as chat_router
from app.routes.websocket import router as websocket_router

app.include_router(health_router, prefix="/api/health", tags=["Health"])
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
app.include_router(websocket_router, prefix="/ws", tags=["WebSocket"])

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with demo information"""
    providers = demo_settings.get_available_providers()
    return {
        "message": "🤖 AI Chatbot Demo API",
        "status": "ready",
        "version": "1.0.0-demo",
        "features": {
            "providers": providers,
            "default_model": demo_settings.get_default_model(),
            "streaming": demo_settings.feature_streaming,
            "websocket": demo_settings.feature_websocket,
            "caching": demo_settings.feature_semantic_cache
        },
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/api/health/ready",
            "chat": "/api/chat",
            "websocket": "/ws/chat"
        },
        "ui": "http://localhost:3000"
    }

@app.get("/api/models", tags=["Models"])
async def get_available_models():
    """Get list of available models"""
    models = []
    
    if demo_settings.openai_api_key:
        models.extend([
            {"provider": "openai", "model": "gpt-3.5-turbo", "description": "Fast and efficient"},
            {"provider": "openai", "model": "gpt-4", "description": "Most capable"},
        ])
    
    if demo_settings.anthropic_api_key:
        models.extend([
            {"provider": "anthropic", "model": "claude-3-haiku-20240307", "description": "Fast responses"},
            {"provider": "anthropic", "model": "claude-3-sonnet-20240229", "description": "Balanced performance"},
        ])
    
    return {
        "models": models,
        "default": demo_settings.get_default_model()
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler for demo"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "tip": "Check the API documentation at /docs for usage examples"
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle configuration errors"""
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "tip": "Please check your .env configuration file"
        }
    )

# Demo-specific endpoints
@app.get("/api/demo/status", tags=["Demo"])
async def demo_status():
    """Get demo system status"""
    return {
        "api": "operational",
        "cache": "operational" if hasattr(app.state, 'cache') and app.state.cache else "disabled",
        "providers": demo_settings.get_available_providers(),
        "rate_limit": {
            "requests_per_minute": demo_settings.rate_limit_requests,
            "enabled": True
        }
    }

@app.post("/api/demo/test", tags=["Demo"])
async def test_chat():
    """Test endpoint to verify LLM connectivity"""
    try:
        # Simple test with the default provider
        from app.providers.openai_adapter import OpenAIAdapter
        
        if demo_settings.openai_api_key:
            adapter = OpenAIAdapter(api_key=demo_settings.openai_api_key)
            response = await adapter.generate_response(
                messages=[{"role": "user", "content": "Say 'Hello, Demo!' in 5 words or less"}],
                model=demo_settings.openai_default_model
            )
            return {
                "success": True,
                "provider": "openai",
                "response": response.get("content", "Test successful")
            }
        else:
            return {
                "success": False,
                "error": "No API keys configured"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main_demo:app",
        host="0.0.0.0",
        port=demo_settings.app_port,
        reload=True,
        log_level=demo_settings.log_level.lower()
    )