"""
Simplified configuration for demo deployment
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional

class DemoSettings(BaseSettings):
    """Demo-optimized settings with sensible defaults"""
    
    # Required API Keys (at least one)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Application Settings
    app_env: str = "demo"
    app_port: int = 8000
    frontend_url: str = "http://localhost:3000"
    debug: bool = False
    
    # API Configuration
    api_title: str = "AI Chatbot Demo"
    api_version: str = "1.0.0-demo"
    
    # Database & Cache
    database_url: str = "postgresql://demo:demo123@postgres:5432/chatbot_demo"
    redis_url: str = "redis://redis:6379/0"
    redis_ttl: int = 3600  # 1 hour cache
    
    # Model Settings
    default_model_provider: str = "openai"
    model_fallback_enabled: bool = True
    openai_default_model: str = "gpt-3.5-turbo"
    anthropic_default_model: str = "claude-3-haiku-20240307"
    max_tokens: int = 1000
    temperature: float = 0.7
    
    # Features
    feature_semantic_cache: bool = True
    feature_streaming: bool = True
    feature_websocket: bool = True
    
    # Rate Limiting (Demo limits)
    rate_limit_requests: int = 30
    rate_limit_window: int = 60
    max_conversation_length: int = 20
    
    # Security
    jwt_secret: str = "demo-secret-key-change-in-production"
    api_key_header: str = "X-API-Key"
    
    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def validate_config(self) -> tuple[bool, str]:
        """
        Validate configuration for demo
        Returns: (is_valid, error_message)
        """
        if not self.openai_api_key and not self.anthropic_api_key:
            return False, "At least one API key required (OPENAI_API_KEY or ANTHROPIC_API_KEY)"
        
        if self.openai_api_key and not self.openai_api_key.startswith("sk-"):
            return False, "Invalid OpenAI API key format"
            
        if self.anthropic_api_key and not self.anthropic_api_key.startswith("sk-"):
            if not self.anthropic_api_key.startswith("anthropic-"):
                return False, "Invalid Anthropic API key format"
        
        return True, ""
    
    def get_available_providers(self) -> list[str]:
        """Get list of configured providers"""
        providers = []
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        return providers
    
    def get_default_model(self) -> str:
        """Get the default model based on available providers"""
        if self.default_model_provider == "openai" and self.openai_api_key:
            return self.openai_default_model
        elif self.default_model_provider == "anthropic" and self.anthropic_api_key:
            return self.anthropic_default_model
        elif self.openai_api_key:
            return self.openai_default_model
        elif self.anthropic_api_key:
            return self.anthropic_default_model
        return "gpt-3.5-turbo"  # Fallback

# Create singleton instance
demo_settings = DemoSettings()