from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal, Union
from datetime import datetime
import uuid

class ImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: Dict[str, str]  # {"url": "data:image/jpeg;base64,..." or "https://..."}

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class Message(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    content: Union[str, List[Union[TextContent, ImageContent]]]
    role: Literal["user", "assistant", "system"]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict] = None

class ChatSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: Literal["active", "ended", "error"] = "active"
    config: Dict = {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    metrics: Dict = {
        "total_messages": 0,
        "total_tokens": 0,
        "total_cost": 0.0
    }

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    images: Optional[List[str]] = None  # Base64 encoded images or URLs

class ChatResponse(BaseModel):
    message: Message
    session_id: str
    tokens_used: int
    cost: float
    model_used: str
    latency_ms: float