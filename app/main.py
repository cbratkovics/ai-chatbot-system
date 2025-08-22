"""Main application entry point."""
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="AI Chatbot System", version="1.0.0")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "AI Chatbot System API"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse(
        status_code=200,
        content={"status": "healthy", "service": "ai-chatbot-system"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
