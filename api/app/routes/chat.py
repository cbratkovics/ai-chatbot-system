from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from app.models.chat import ChatRequest, ChatResponse, ChatSession, Message
from app.services.session.session_manager import SessionManager
from app.services.llm.openai_provider import OpenAIProvider
from app.config import settings
import time

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# Initialize services
session_manager = SessionManager()
llm_provider = OpenAIProvider()

@router.post("/sessions", response_model=ChatSession)
async def create_session(user_id: Optional[str] = None):
    """Create a new chat session"""
    session = await session_manager.create_session(user_id)
    return session

@router.get("/sessions/{session_id}", response_model=ChatSession)
async def get_session(session_id: str):
    """Get session details"""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.post("/messages", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """Send a message and get AI response"""
    start_time = time.time()
    
    # Get or create session
    if request.session_id:
        session = await session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = await session_manager.create_session()
    
    # Add user message
    user_message = Message(
        session_id=session.session_id,
        content=request.message,
        role="user"
    )
    await session_manager.add_message(session.session_id, user_message)
    
    # Get conversation history
    messages = await session_manager.get_messages(session.session_id)
    
    # Generate AI response
    model = request.model or session.config.get("model", settings.default_model)
    temperature = request.temperature or session.config.get("temperature", settings.temperature)
    max_tokens = request.max_tokens or session.config.get("max_tokens", settings.max_tokens)
    
    response = await llm_provider.generate_response(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Create assistant message
    assistant_message = Message(
        session_id=session.session_id,
        content=response["content"],
        role="assistant",
        metadata={
            "model": response["model"],
            "tokens": response["total_tokens"],
            "cost": response["cost"]
        }
    )
    await session_manager.add_message(session.session_id, assistant_message)
    
    # Update session metrics
    session.metrics["total_messages"] += 2
    session.metrics["total_tokens"] += response["total_tokens"]
    session.metrics["total_cost"] += response["cost"]
    await session_manager.update_session(session)
    
    return ChatResponse(
        message=assistant_message,
        session_id=session.session_id,
        tokens_used=response["total_tokens"],
        cost=response["cost"],
        model_used=response["model"],
        latency_ms=(time.time() - start_time) * 1000
    )

@router.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str, limit: int = 50):
    """Get message history for a session"""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = await session_manager.get_messages(session_id, limit)
    return messages