from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    api_title: str = "AI Conversational Platform"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    redis_ttl: int = 86400  # 24 hours
    
    # LLM Configuration
    openai_api_key: str
    anthropic_api_key: Optional[str] = None
    default_model: str = "gpt-4"
    max_tokens: int = 1000
    temperature: float = 0.7
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()