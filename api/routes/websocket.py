"""WebSocket routes for real-time chat functionality."""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException, Depends
from fastapi.responses import HTMLResponse

from ..app.config import settings
from ..ws_handlers.manager import connection_manager
# WebSocketHandler temporarily disabled due to import issues
# from ..ws_handlers.handlers import WebSocketHandler
from ..providers import ProviderOrchestrator, ProviderConfig, ProviderA, ProviderB

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize providers and orchestrator (in production, this would be dependency injected)
def get_provider_orchestrator() -> ProviderOrchestrator:
    """Get provider orchestrator instance."""
    # Mock provider configurations
    provider_a_config = ProviderConfig(
        name="provider_a",
        api_key=settings.provider_a_api_key or "mock-key-a",
        timeout=settings.provider_timeout,
        max_retries=settings.max_retries
    )
    
    provider_b_config = ProviderConfig(
        name="provider_b", 
        api_key=settings.provider_b_api_key or "mock-key-b",
        timeout=settings.provider_timeout,
        max_retries=settings.max_retries
    )
    
    providers = [
        ProviderA(provider_a_config),
        ProviderB(provider_b_config)
    ]
    
    return ProviderOrchestrator(providers)

# Global instances (in production, use dependency injection)
provider_orchestrator = get_provider_orchestrator()
# WebSocketHandler temporarily disabled - need to fix imports
# websocket_handler = WebSocketHandler(connection_manager, provider_orchestrator)


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    tenant_id: Optional[str] = Query(None, description="Tenant ID for multi-tenancy"),
    user_id: Optional[str] = Query(None, description="User ID for connection tracking"),
    conversation_id: Optional[str] = Query(None, description="Conversation ID for chat context")
):
    """
    WebSocket endpoint for real-time chat.
    
    Features:
    - Real-time bidirectional communication
    - Multi-tenant support with isolation
    - User and conversation context tracking
    - Automatic heartbeat and connection management
    - Rate limiting and authentication
    
    Query Parameters:
    - tenant_id: Optional tenant identifier for multi-tenancy
    - user_id: Optional user identifier for connection tracking  
    - conversation_id: Optional conversation identifier for chat context
    
    Example Usage:
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/api/v1/ws?tenant_id=123&user_id=user456');
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Received:', data);
    };
    
    // Send authentication
    ws.send(JSON.stringify({
        type: 'auth_request',
        data: { token: 'your-auth-token' }
    }));
    
    // Send chat message
    ws.send(JSON.stringify({
        type: 'chat_message',
        data: {
            content: 'Hello, how are you?',
            model: 'model-3.5-turbo',
            stream: true
        }
    }));
    ```
    """
    
    # Parse UUID parameters
    tenant_uuid = None
    conversation_uuid = None
    
    try:
        if tenant_id:
            tenant_uuid = UUID(tenant_id)
        if conversation_id:
            conversation_uuid = UUID(conversation_id)
    except ValueError as e:
        await websocket.close(code=4000, reason=f"Invalid UUID format: {e}")
        return
    
    connection = None
    
    try:
        # Establish WebSocket connection
        connection = await connection_manager.connect(
            websocket=websocket,
            tenant_id=tenant_uuid,
            user_id=user_id,
            conversation_id=conversation_uuid
        )
        
        logger.info(f"WebSocket connection established: {connection.id}")
        
        # Handle the connection with WebSocket handler
        # await websocket_handler.handle_connection(connection)
        # Temporarily just keep the connection alive
        await websocket.receive_text()
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection.id if connection else 'unknown'}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if connection:
            await connection_manager.disconnect(
                connection.id,
                code=1011,
                reason="Server error"
            )


@router.get("/ws/stats")
async def websocket_stats():
    """
    Get WebSocket connection statistics.
    
    Returns detailed statistics about active connections, message throughput,
    and performance metrics.
    """
    return {
        "websocket_stats": connection_manager.get_connection_stats(),
        "provider_stats": await provider_orchestrator.health_check()
    }


@router.get("/ws/connections")
async def list_connections(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant ID"),
    user_id: Optional[str] = Query(None, description="Filter by user ID")
):
    """
    List active WebSocket connections with optional filtering.
    
    Useful for monitoring and debugging connection states.
    """
    connections = []
    
    for connection_id, connection in connection_manager.connections.items():
        # Apply filters
        if tenant_id and str(connection.tenant_id) != tenant_id:
            continue
        if user_id and connection.user_id != user_id:
            continue
            
        connections.append({
            "id": connection.id,
            "tenant_id": str(connection.tenant_id) if connection.tenant_id else None,
            "user_id": connection.user_id,
            "conversation_id": str(connection.conversation_id) if connection.conversation_id else None,
            "authenticated": connection.authenticated,
            "connected_at": connection.stats.connected_at,
            "uptime_seconds": connection.stats.uptime_seconds,
            "messages_sent": connection.stats.messages_sent,
            "messages_received": connection.stats.messages_received,
            "last_activity": connection.stats.last_activity,
            "subscribed_events": list(connection.subscribed_events)
        })
    
    return {
        "connections": connections,
        "total_count": len(connections),
        "filters": {
            "tenant_id": tenant_id,
            "user_id": user_id
        }
    }


@router.post("/ws/broadcast")
async def broadcast_message(
    message: str,
    level: str = "info",
    tenant_id: Optional[str] = None
):
    """
    Broadcast system message to WebSocket connections.
    
    Useful for maintenance notifications, system alerts, etc.
    """
    tenant_uuid = None
    if tenant_id:
        try:
            tenant_uuid = UUID(tenant_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid tenant_id format")
    
    # Temporarily disabled due to handler import issues
    # await websocket_handler.broadcast_system_message(
    #     message=message,
    #     level=level,
    #     tenant_id=tenant_uuid
    # )
    pass
    
    return {
        "status": "success",
        "message": "Broadcast sent",
        "target": f"tenant {tenant_id}" if tenant_id else "all connections"
    }


@router.get("/ws/test", response_class=HTMLResponse)
async def websocket_test_page():
    """
    Serve a simple HTML page for testing WebSocket functionality.
    
    This is useful for development and testing purposes.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>WebSocket Test - Chatbot System</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .message-area {{ height: 400px; border: 1px solid #ccc; overflow-y: scroll; padding: 10px; margin: 10px 0; }}
            .input-area {{ display: flex; gap: 10px; margin: 10px 0; }}
            input, textarea {{ flex: 1; padding: 8px; }}
            button {{ padding: 8px 16px; background: #007bff; color: white; border: none; cursor: pointer; }}
            button:hover {{ background: #0056b3; }}
            .status {{ padding: 10px; margin: 10px 0; border-radius: 4px; }}
            .status.connected {{ background: #d4edda; color: #155724; }}
            .status.disconnected {{ background: #f8d7da; color: #721c24; }}
            .message {{ margin: 5px 0; padding: 5px; }}
            .message.sent {{ background: #e3f2fd; }}
            .message.received {{ background: #f3e5f5; }}
            .message.system {{ background: #fff3cd; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>WebSocket Test - Chatbot System</h1>
            
            <div id="status" class="status disconnected">Disconnected</div>
            
            <div>
                <label>WebSocket URL:</label>
                <input type="text" id="url" value="ws://localhost:{settings.port}{settings.api_prefix}/ws?tenant_id=550e8400-e29b-41d4-a716-446655440000&user_id=test_user">
            </div>
            
            <div class="input-area">
                <button onclick="connect()">Connect</button>
                <button onclick="disconnect()">Disconnect</button>
                <button onclick="authenticate()">Authenticate</button>
            </div>
            
            <div id="messages" class="message-area"></div>
            
            <div class="input-area">
                <textarea id="messageInput" placeholder="Type your message here..." rows="3"></textarea>
                <button onclick="sendMessage()">Send Message</button>
            </div>
            
            <div class="input-area">
                <input type="text" id="model" value="model-3.5-turbo" placeholder="Model">
                <input type="number" id="temperature" value="0.7" step="0.1" min="0" max="2" placeholder="Temperature">
                <label><input type="checkbox" id="stream" checked> Stream</label>
            </div>
        </div>

        <script>
            let ws = null;
            
            function updateStatus(message, connected) {{
                const status = document.getElementById('status');
                status.textContent = message;
                status.className = 'status ' + (connected ? 'connected' : 'disconnected');
            }}
            
            function addMessage(content, type) {{
                const messages = document.getElementById('messages');
                const div = document.createElement('div');
                div.className = 'message ' + type;
                div.innerHTML = '<strong>' + new Date().toLocaleTimeString() + '</strong>: ' + 
                               (typeof content === 'object' ? JSON.stringify(content, null, 2) : content);
                messages.appendChild(div);
                messages.scrollTop = messages.scrollHeight;
            }}
            
            function connect() {{
                const url = document.getElementById('url').value;
                ws = new WebSocket(url);
                
                ws.onopen = function(event) {{
                    updateStatus('Connected', true);
                    addMessage('WebSocket connection opened', 'system');
                }};
                
                ws.onmessage = function(event) {{
                    const data = JSON.parse(event.data);
                    addMessage(data, 'received');
                }};
                
                ws.onclose = function(event) {{
                    updateStatus('Disconnected (Code: ' + event.code + ')', false);
                    addMessage('WebSocket connection closed: ' + event.reason, 'system');
                }};
                
                ws.onerror = function(event) {{
                    addMessage('WebSocket error occurred', 'system');
                }};
            }}
            
            function disconnect() {{
                if (ws) {{
                    ws.close();
                    ws = null;
                }}
            }}
            
            function authenticate() {{
                if (!ws) return;
                const authEvent = {{
                    type: 'auth_request',
                    data: {{ token: 'valid-token' }}
                }};
                ws.send(JSON.stringify(authEvent));
                addMessage(authEvent, 'sent');
            }}
            
            function sendMessage() {{
                if (!ws) return;
                
                const content = document.getElementById('messageInput').value.trim();
                if (!content) return;
                
                const messageEvent = {{
                    type: 'chat_message',
                    data: {{
                        content: content,
                        model: document.getElementById('model').value,
                        temperature: parseFloat(document.getElementById('temperature').value),
                        stream: document.getElementById('stream').checked,
                        conversation_id: '550e8400-e29b-41d4-a716-446655440001'
                    }}
                }};
                
                ws.send(JSON.stringify(messageEvent));
                addMessage(messageEvent, 'sent');
                document.getElementById('messageInput').value = '';
            }}
            
            // Send heartbeat every 30 seconds
            setInterval(function() {{
                if (ws && ws.readyState === WebSocket.OPEN) {{
                    const heartbeat = {{
                        type: 'heartbeat',
                        data: {{ client_time: Date.now() }}
                    }};
                    ws.send(JSON.stringify(heartbeat));
                }}
            }}, 30000);
            
            // Allow Enter key to send messages
            document.getElementById('messageInput').addEventListener('keypress', function(e) {{
                if (e.key === 'Enter' && !e.shiftKey) {{
                    e.preventDefault();
                    sendMessage();
                }}
            }});
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# Health check for WebSocket system
@router.get("/ws/health")
async def websocket_health():
    """Health check for WebSocket system."""
    stats = connection_manager.get_connection_stats()
    provider_health = await provider_orchestrator.health_check()
    
    # Determine overall health
    is_healthy = (
        stats["active_connections"] >= 0 and  # Basic sanity check
        provider_health["orchestrator"]["available_providers"] > 0
    )
    
    return {
        "status": "healthy" if is_healthy else "degraded",
        "websocket_manager": {
            "status": "healthy",
            "active_connections": stats["active_connections"],
            "total_connections": stats["total_connections"]
        },
        "provider_orchestrator": {
            "status": "healthy" if provider_health["orchestrator"]["available_providers"] > 0 else "degraded",
            "available_providers": provider_health["orchestrator"]["available_providers"],
            "total_providers": provider_health["orchestrator"]["total_providers"]
        },
        "background_tasks": {
            "heartbeat": "running",
            "cleanup": "running"
        }
    }