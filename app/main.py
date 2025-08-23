"""Main FastAPI application."""
from fastapi import FastAPI

app = FastAPI(
    title="System API",
    description="Core system API for chatbot functionality",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "api": "operational",
            "database": "connected"
        }
    }


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "System API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/info")
def get_info():
    """Get system information."""
    return {
        "system": "Chatbot System",
        "framework": "FastAPI",
        "python_version": "3.11+"
    }
