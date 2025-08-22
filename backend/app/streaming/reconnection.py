import asyncio
from typing import Optional, Callable, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ReconnectionManager:
    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        
        self.connections: Dict[str, Dict[str, Any]] = {}
        
    def register_connection(self, connection_id: str, reconnect_callback: Callable):
        self.connections[connection_id] = {
            "callback": reconnect_callback,
            "retry_count": 0,
            "last_attempt": None,
            "is_connected": False,
            "created_at": datetime.now()
        }
        logger.info(f"Registered connection: {connection_id}")
    
    def unregister_connection(self, connection_id: str):
        if connection_id in self.connections:
            del self.connections[connection_id]
            logger.info(f"Unregistered connection: {connection_id}")
    
    async def handle_disconnect(self, connection_id: str):
        if connection_id not in self.connections:
            logger.warning(f"Unknown connection: {connection_id}")
            return
        
        conn_info = self.connections[connection_id]
        conn_info["is_connected"] = False
        
        await self._attempt_reconnection(connection_id)
    
    async def _attempt_reconnection(self, connection_id: str):
        if connection_id not in self.connections:
            return
        
        conn_info = self.connections[connection_id]
        
        while conn_info["retry_count"] < self.max_retries:
            conn_info["retry_count"] += 1
            conn_info["last_attempt"] = datetime.now()
            
            # Calculate delay with exponential backoff
            delay = min(
                self.base_delay * (self.exponential_base ** (conn_info["retry_count"] - 1)),
                self.max_delay
            )
            
            logger.info(f"Attempting reconnection {conn_info['retry_count']}/{self.max_retries} "
                       f"for {connection_id} in {delay:.1f}s")
            
            await asyncio.sleep(delay)
            
            try:
                # Call the reconnection callback
                await conn_info["callback"]()
                
                # Success - reset retry count
                conn_info["retry_count"] = 0
                conn_info["is_connected"] = True
                logger.info(f"Successfully reconnected: {connection_id}")
                return
                
            except Exception as e:
                logger.error(f"Reconnection attempt failed for {connection_id}: {e}")
                
                if conn_info["retry_count"] >= self.max_retries:
                    logger.error(f"Max retries reached for {connection_id}. Giving up.")
                    self.unregister_connection(connection_id)
                    return
    
    def mark_connected(self, connection_id: str):
        if connection_id in self.connections:
            self.connections[connection_id]["is_connected"] = True
            self.connections[connection_id]["retry_count"] = 0
    
    def is_connected(self, connection_id: str) -> bool:
        if connection_id in self.connections:
            return self.connections[connection_id]["is_connected"]
        return False
    
    def get_connection_stats(self, connection_id: str) -> Optional[Dict[str, Any]]:
        if connection_id not in self.connections:
            return None
        
        conn_info = self.connections[connection_id]
        return {
            "connection_id": connection_id,
            "is_connected": conn_info["is_connected"],
            "retry_count": conn_info["retry_count"],
            "last_attempt": conn_info["last_attempt"],
            "created_at": conn_info["created_at"],
            "uptime": datetime.now() - conn_info["created_at"] if conn_info["is_connected"] else None
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        total_connections = len(self.connections)
        connected_count = sum(1 for c in self.connections.values() if c["is_connected"])
        
        return {
            "total_connections": total_connections,
            "connected": connected_count,
            "disconnected": total_connections - connected_count,
            "connections": {
                conn_id: self.get_connection_stats(conn_id)
                for conn_id in self.connections
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        healthy_connections = []
        unhealthy_connections = []
        
        for conn_id, conn_info in self.connections.items():
            if conn_info["is_connected"]:
                healthy_connections.append(conn_id)
            else:
                unhealthy_connections.append({
                    "id": conn_id,
                    "retry_count": conn_info["retry_count"],
                    "last_attempt": conn_info["last_attempt"]
                })
        
        return {
            "healthy": len(healthy_connections),
            "unhealthy": len(unhealthy_connections),
            "unhealthy_details": unhealthy_connections
        }