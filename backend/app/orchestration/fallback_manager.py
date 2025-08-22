from typing import List, Optional, Dict, Any, AsyncGenerator
import asyncio
import logging
from datetime import datetime

from ..providers.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class FallbackManager:
    def __init__(self, primary_providers: List[BaseProvider], fallback_providers: List[BaseProvider]):
        self.primary_providers = primary_providers
        self.fallback_providers = fallback_providers
        self.failure_counts: Dict[str, int] = {}
        self.last_failure_times: Dict[str, datetime] = {}
        self.max_retries = 3
        self.fallback_threshold = 2

    async def execute_with_fallback(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        # Try primary providers first
        for provider in self.primary_providers:
            try:
                logger.info(f"Attempting request with primary provider: {provider.name}")
                async for chunk in provider.stream_completion(messages, **kwargs):
                    yield chunk
                
                # Reset failure count on success
                self.failure_counts[provider.name] = 0
                return
                
            except Exception as e:
                logger.warning(f"Primary provider {provider.name} failed: {e}")
                self._record_failure(provider.name)
                
                if self._should_use_fallback(provider.name):
                    break
        
        # Use fallback providers
        for provider in self.fallback_providers:
            try:
                logger.info(f"Attempting request with fallback provider: {provider.name}")
                async for chunk in provider.stream_completion(messages, **kwargs):
                    yield chunk
                return
                
            except Exception as e:
                logger.error(f"Fallback provider {provider.name} failed: {e}")
                continue
        
        # All providers failed
        raise Exception("All providers (primary and fallback) failed to process the request")

    def _record_failure(self, provider_name: str):
        self.failure_counts[provider_name] = self.failure_counts.get(provider_name, 0) + 1
        self.last_failure_times[provider_name] = datetime.now()

    def _should_use_fallback(self, provider_name: str) -> bool:
        failure_count = self.failure_counts.get(provider_name, 0)
        return failure_count >= self.fallback_threshold

    async def health_check(self) -> Dict[str, Any]:
        health_status = {
            "primary_providers": {},
            "fallback_providers": {},
            "overall_health": "healthy"
        }
        
        # Check primary providers
        for provider in self.primary_providers:
            failures = self.failure_counts.get(provider.name, 0)
            health_status["primary_providers"][provider.name] = {
                "status": "healthy" if failures < self.fallback_threshold else "degraded",
                "failure_count": failures,
                "last_failure": self.last_failure_times.get(provider.name)
            }
        
        # Check fallback providers  
        for provider in self.fallback_providers:
            failures = self.failure_counts.get(provider.name, 0)
            health_status["fallback_providers"][provider.name] = {
                "status": "healthy" if failures == 0 else "degraded",
                "failure_count": failures,
                "last_failure": self.last_failure_times.get(provider.name)
            }
        
        # Determine overall health
        primary_healthy = any(
            self.failure_counts.get(p.name, 0) < self.fallback_threshold 
            for p in self.primary_providers
        )
        
        if not primary_healthy:
            fallback_healthy = any(
                self.failure_counts.get(p.name, 0) == 0 
                for p in self.fallback_providers
            )
            health_status["overall_health"] = "degraded" if fallback_healthy else "critical"
        
        return health_status

    def reset_provider(self, provider_name: str):
        self.failure_counts[provider_name] = 0
        if provider_name in self.last_failure_times:
            del self.last_failure_times[provider_name]
        logger.info(f"Reset failure counts for provider: {provider_name}")