from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from app.models.chat import ChatRequest, ChatResponse, ChatSession, Message
from app.services.session.session_manager import SessionManager
from app.services.llm.openai_provider import OpenAIProvider
from app.services.llm.orchestrator import LLMOrchestrator
from app.services.cache.semantic_cache import SemanticCache
from app.services.monitoring.metrics import (
    tokens_used, api_cost, model_latency, active_sessions,
    cache_hits, cache_misses, track_metrics
)
from app.services.monitoring.quality import QualityEvaluator
from app.services.image_processor import ImageProcessor
from app.models.chat import TextContent, ImageContent
from app.config import settings
import time
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# Initialize services
session_manager = SessionManager()
llm_provider = OpenAIProvider()
orchestrator = LLMOrchestrator()
semantic_cache = SemanticCache()
quality_evaluator = QualityEvaluator()
image_processor = ImageProcessor()

@router.post("/sessions", response_model=ChatSession)
async def create_session(user_id: Optional[str] = None):
    """Create a new chat session"""
    session = await session_manager.create_session(user_id)
    active_sessions.inc()
    return session

@router.get("/sessions/{session_id}", response_model=ChatSession)
async def get_session(session_id: str):
    """Get session details"""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.post("/messages", response_model=ChatResponse)
@track_metrics("/messages")
async def send_message(request: ChatRequest):
    """Enhanced send message with caching and monitoring"""
    start_time = time.time()
    
    # Get or create session
    if request.session_id:
        session = await session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = await session_manager.create_session()
        active_sessions.inc()
    
    # Check cache first
    cached_response = await semantic_cache.get_cached_response(request.message)
    if cached_response:
        cache_hits.labels(cache_type="semantic").inc()
        response_text, similarity = cached_response
        
        # Create cached response message
        assistant_message = Message(
            session_id=session.session_id,
            content=response_text,
            role="assistant",
            metadata={
                "cached": True,
                "similarity": similarity
            }
        )
        await session_manager.add_message(session.session_id, assistant_message)
        
        return ChatResponse(
            message=assistant_message,
            session_id=session.session_id,
            tokens_used=0,
            cost=0.0,
            model_used="cached",
            latency_ms=(time.time() - start_time) * 1000
        )
    else:
        cache_misses.labels(cache_type="semantic").inc()
    
    # Process images if provided
    message_content = request.message
    if request.images:
        # Create multi-modal content
        content_parts = [TextContent(text=request.message)]
        
        for image_data in request.images[:4]:  # Limit to 4 images
            try:
                processed_image = await image_processor.process_image(image_data)
                content_parts.append(ImageContent(
                    image_url={"url": processed_image}
                ))
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
        
        message_content = content_parts
    
    # Add user message
    user_message = Message(
        session_id=session.session_id,
        content=message_content,
        role="user"
    )
    await session_manager.add_message(session.session_id, user_message)
    
    # Get conversation history
    messages = await session_manager.get_messages(session.session_id)
    
    # Use orchestrator for intelligent model selection
    selected_model = orchestrator.select_model_for_query(
        request.message,
        optimize_cost=True
    )
    
    # Generate AI response with failover
    llm_start = time.time()
    response = await orchestrator.generate_response_with_failover(
        messages=messages,
        preferred_model=request.model or selected_model,
        temperature=request.temperature or session.config.get("temperature", settings.temperature),
        max_tokens=request.max_tokens or session.config.get("max_tokens", settings.max_tokens)
    )
    llm_duration = time.time() - llm_start
    
    # Track metrics
    tokens_used.labels(
        model=response["model"],
        provider=response["provider"]
    ).inc(response["total_tokens"])
    
    api_cost.labels(
        model=response["model"],
        provider=response["provider"]
    ).inc(response["cost"])
    
    model_latency.labels(
        model=response["model"],
        provider=response["provider"]
    ).observe(llm_duration)
    
    # Cache the response
    await semantic_cache.cache_response(request.message, response["content"])
    
    # Evaluate response quality
    quality_scores = await quality_evaluator.evaluate_response(
        query=request.message,
        response=response["content"]
    )
    
    # Create assistant message
    assistant_message = Message(
        session_id=session.session_id,
        content=response["content"],
        role="assistant",
        metadata={
            "model": response["model"],
            "provider": response["provider"],
            "tokens": response["total_tokens"],
            "cost": response["cost"],
            "quality_scores": quality_scores,
            "failover": response.get("failover", False)
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