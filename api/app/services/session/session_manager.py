import redis.asyncio as redis
import json
from typing import List, Optional
from datetime import datetime
from app.models.chat import ChatSession, Message
from app.config import settings

class SessionManager:
    def __init__(self):
        self.redis = redis.from_url(settings.redis_url, decode_responses=True)
    
    async def create_session(self, user_id: Optional[str] = None) -> ChatSession:
        session = ChatSession(user_id=user_id)
        
        # Store session data
        session_key = f"session:{session.session_id}"
        await self.redis.hset(
            session_key,
            mapping={
                "data": session.model_dump_json(),
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat()
            }
        )
        
        # Set TTL
        await self.redis.expire(session_key, settings.redis_ttl)
        
        # Add to user's session set if user_id provided
        if user_id:
            await self.redis.sadd(f"user_sessions:{user_id}", session.session_id)
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        session_key = f"session:{session_id}"
        session_data = await self.redis.hget(session_key, "data")
        
        if not session_data:
            return None
        
        return ChatSession.model_validate_json(session_data)
    
    async def update_session(self, session: ChatSession):
        session.updated_at = datetime.utcnow()
        session_key = f"session:{session.session_id}"
        
        await self.redis.hset(
            session_key,
            mapping={
                "data": session.model_dump_json(),
                "updated_at": session.updated_at.isoformat()
            }
        )
        
        # Reset TTL
        await self.redis.expire(session_key, settings.redis_ttl)
    
    async def add_message(self, session_id: str, message: Message):
        messages_key = f"messages:{session_id}"
        await self.redis.rpush(messages_key, message.model_dump_json())
        await self.redis.expire(messages_key, settings.redis_ttl)
    
    async def get_messages(self, session_id: str, limit: int = 50) -> List[Message]:
        messages_key = f"messages:{session_id}"
        
        # Get last 'limit' messages
        messages_json = await self.redis.lrange(messages_key, -limit, -1)
        
        return [
            Message.model_validate_json(msg_json)
            for msg_json in messages_json
        ]
    
    async def close(self):
        await self.redis.close()