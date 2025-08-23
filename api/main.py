"""Main FastAPI application for Enterprise Chatbot System."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import chat, health, websocket
from .middleware import auth, rate_limiter, error_handler
from .config import settings

app = FastAPI(
    title="Enterprise Chatbot System",
    version="1.0.0",
    description="Production-grade multi-provider chatbot platform"
)

# Middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"])
app.add_middleware(error_handler.ErrorHandlerMiddleware)
app.add_middleware(rate_limiter.RateLimitMiddleware)
app.add_middleware(auth.AuthMiddleware)

# Routes
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(websocket.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
