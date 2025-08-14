from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import datetime

class SessionMetrics(BaseModel):
    session_id: str
    total_messages: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_response_time_ms: float = 0.0
    quality_scores: Dict[str, float] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SystemMetrics(BaseModel):
    total_sessions: int = 0
    active_sessions: int = 0
    total_requests: int = 0
    cache_hit_rate: float = 0.0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)