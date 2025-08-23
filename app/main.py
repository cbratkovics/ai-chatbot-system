"""Main FastAPI application - Moved to api/app/main.py for proper structure."""

# This file exists for backward compatibility
# The actual application is now located at api/app/main.py

from fastapi import FastAPI

app = FastAPI(
    title="AI Chatbot System (Legacy Endpoint)",
    description="Please use the main API at /api/v1",
    version="1.0.0"
)

@app.get("/")
def root():
    """Legacy root endpoint - redirects to main API."""
    return {
        "message": "This is a legacy endpoint. Please use the main API at /api/v1",
        "main_api": "/api/v1",
        "docs": "/api/v1/docs",
        "health": "/api/v1/health"
    }
