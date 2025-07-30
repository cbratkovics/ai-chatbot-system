from fastapi import WebSocket
from typing import Dict, Set
import json
import asyncio

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.connection_lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        async with self.connection_lock:
            if session_id not in self.active_connections:
                self.active_connections[session_id] = set()
            self.active_connections[session_id].add(websocket)
    
    async def disconnect(self, websocket: WebSocket, session_id: str):
        async with self.connection_lock:
            if session_id in self.active_connections:
                self.active_connections[session_id].discard(websocket)
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            disconnected = set()
            for websocket in self.active_connections[session_id]:
                try:
                    await websocket.send_json(message)
                except:
                    disconnected.add(websocket)
            
            # Clean up disconnected websockets
            async with self.connection_lock:
                for ws in disconnected:
                    self.active_connections[session_id].discard(ws)

manager = ConnectionManager()