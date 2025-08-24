"""Application configuration management."""
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Application
    APP_NAME: str = "Enterprise Chatbot Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    PORT: int = 8000
    WORKERS: int = 4
    LOG_LEVEL: str = "INFO"

    # Security
    SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 30
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Database
    DATABASE_URL: str
    DATABASE_URL_TEST: str = ""

    # Redis
    REDIS_URL: str
    CACHE_TTL: int = 3600

    # Providers
    PROVIDER_A_API_KEY: str = ""
    PROVIDER_A_BASE_URL: str = ""
    PROVIDER_B_API_KEY: str = ""
    PROVIDER_B_BASE_URL: str = ""

    # Monitoring
    JAEGER_AGENT_HOST: str = "localhost"
    JAEGER_AGENT_PORT: int = 6831
    PROMETHEUS_PORT: int = 9090

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
