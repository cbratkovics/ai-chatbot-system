from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.websocket_manager import manager
from app.services.session.session_manager import SessionManager
from app.services.llm.openai_provider import OpenAIProvider
from app.models.chat import Message
import json
import asyncio

router = APIRouter()

session_manager = SessionManager()
llm_provider = OpenAIProvider()

@router.websocket("/api/v1/chat/stream/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Validate session
            session = await session_manager.get_session(session_id)
            if not session:
                await websocket.send_json({
                    "type": "error",
                    "message": "Session not found"
                })
                continue
            
            # Add user message
            user_message = Message(
                session_id=session_id,
                content=message_data["message"],
                role="user"
            )
            await session_manager.add_message(session_id, user_message)
            
            # Send typing indicator
            await websocket.send_json({
                "type": "typing",
                "status": "start"
            })
            
            # Get conversation history
            messages = await session_manager.get_messages(session_id)
            
            # Stream AI response
            full_response = ""
            token_count = 0
            
            async for chunk in llm_provider.stream_response(
                messages=messages,
                model=session.config.get("model", "gpt-4"),
                temperature=session.config.get("temperature", 0.7),
                max_tokens=session.config.get("max_tokens", 1000)
            ):
                full_response += chunk
                token_count += 1
                
                # Send chunk to client
                await websocket.send_json({
                    "type": "stream",
                    "content": chunk,
                    "tokens": token_count
                })
            
            # Save assistant message
            assistant_message = Message(
                session_id=session_id,
                content=full_response,
                role="assistant"
            )
            await session_manager.add_message(session_id, assistant_message)
            
            # Calculate cost (estimation)
            estimated_tokens = llm_provider.estimate_tokens(full_response)
            cost = llm_provider.calculate_cost(
                input_tokens=sum(llm_provider.estimate_tokens(m.content) for m in messages),
                output_tokens=estimated_tokens,
                model=session.config.get("model", "gpt-4")
            )
            
            # Send completion message
            await websocket.send_json({
                "type": "complete",
                "tokens": estimated_tokens,
                "cost": cost
            })
            
            # Update session metrics
            session.metrics["total_messages"] += 2
            session.metrics["total_tokens"] += estimated_tokens
            session.metrics["total_cost"] += cost
            await session_manager.update_session(session)
            
    except WebSocketDisconnect:
        await manager.disconnect(websocket, session_id)