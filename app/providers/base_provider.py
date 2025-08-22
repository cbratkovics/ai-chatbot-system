"""Base provider interface for AI model adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from enum import Enum
import asyncio
from datetime import datetime


class ModelCapability(Enum):
    """Supported model capabilities."""

    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"
    EMBEDDINGS = "embeddings"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    STREAMING = "streaming"


class ProviderStatus(Enum):
    """Provider health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    model_id: str
    max_tokens: int
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class Message:
    """Chat message structure."""

    role: str  # 'system', 'user', 'assistant', 'function'
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CompletionResponse:
    """Response from model completion."""

    content: str
    model: str
    usage: Dict[str, int]  # tokens_prompt, tokens_completion, tokens_total
    finish_reason: str
    latency_ms: float
    cached: bool = False
    provider: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StreamChunk:
    """Streaming response chunk."""

    delta: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProviderMetrics:
    """Provider performance metrics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_rate: float = 0.0
    last_updated: Optional[datetime] = None


class BaseProvider(ABC):
    """Abstract base class for AI model providers."""

    def __init__(self, api_key: str, config: Optional[Dict[str, Any]] = None):
        """Initialize provider with API key and configuration."""
        self.api_key = api_key
        self.config = config or {}
        self.metrics = ProviderMetrics()
        self._status = ProviderStatus.UNKNOWN
        self._circuit_breaker_open = False
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5

    @property
    @abstractmethod
    def name(self) -> str:
        """Return provider name."""
        pass

    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """Return list of supported model IDs."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> List[ModelCapability]:
        """Return list of provider capabilities."""
        pass

    @abstractmethod
    async def complete(self, messages: List[Message], model_config: ModelConfig) -> CompletionResponse:
        """Generate completion for messages."""
        pass

    @abstractmethod
    async def stream_complete(self, messages: List[Message], model_config: ModelConfig) -> AsyncIterator[StreamChunk]:
        """Stream completion for messages."""
        pass

    @abstractmethod
    async def generate_embeddings(self, texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
        """Generate embeddings for texts."""
        pass

    async def health_check(self) -> ProviderStatus:
        """Check provider health status."""
        try:
            # Simple test completion
            test_message = [Message(role="user", content="test")]
            test_config = ModelConfig(model_id=self.supported_models[0], max_tokens=5, temperature=0)
            await asyncio.wait_for(self.complete(test_message, test_config), timeout=5.0)
            self._status = ProviderStatus.HEALTHY
            self._consecutive_failures = 0
            self._circuit_breaker_open = False
        except asyncio.TimeoutError:
            self._status = ProviderStatus.DEGRADED
            self._consecutive_failures += 1
        except Exception:
            self._status = ProviderStatus.UNHEALTHY
            self._consecutive_failures += 1

        # Open circuit breaker if too many failures
        if self._consecutive_failures >= self._max_consecutive_failures:
            self._circuit_breaker_open = True

        return self._status

    def is_available(self) -> bool:
        """Check if provider is available for requests."""
        return not self._circuit_breaker_open and self._status != ProviderStatus.UNHEALTHY

    async def reset_circuit_breaker(self) -> None:
        """Attempt to reset circuit breaker."""
        status = await self.health_check()
        if status == ProviderStatus.HEALTHY:
            self._circuit_breaker_open = False
            self._consecutive_failures = 0

    def update_metrics(self, success: bool, tokens: int, cost: float, latency_ms: float) -> None:
        """Update provider metrics."""
        self.metrics.total_requests += 1
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1

        self.metrics.total_tokens += tokens
        self.metrics.total_cost += cost

        # Update latency (simplified - should use proper percentile calculation)
        if self.metrics.average_latency_ms == 0:
            self.metrics.average_latency_ms = latency_ms
        else:
            self.metrics.average_latency_ms = self.metrics.average_latency_ms * 0.9 + latency_ms * 0.1

        self.metrics.error_rate = self.metrics.failed_requests / self.metrics.total_requests
        self.metrics.last_updated = datetime.utcnow()

    def get_metrics(self) -> ProviderMetrics:
        """Get current provider metrics."""
        return self.metrics

    @abstractmethod
    def estimate_cost(self, tokens_prompt: int, tokens_completion: int, model: str) -> float:
        """Estimate cost for token usage."""
        pass

    def validate_model(self, model_id: str) -> bool:
        """Validate if model is supported."""
        return model_id in self.supported_models

    def __repr__(self) -> str:
        """String representation of provider."""
        return f"{self.name}(status={self._status.value}, models={len(self.supported_models)})"
