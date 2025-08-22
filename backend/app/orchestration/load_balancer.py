from typing import List, Optional, Dict, Any
import random
import asyncio
from datetime import datetime, timedelta
import logging

from ..providers.base_provider import BaseProvider

logger = logging.getLogger(__name__)


class LoadBalancer:
    def __init__(self, providers: List[BaseProvider]):
        self.providers = providers
        self.provider_stats: Dict[str, Dict[str, Any]] = {}
        self.strategy = "round_robin"  # Options: round_robin, least_connections, weighted, latency_based
        self.current_index = 0
        self._init_stats()

    def _init_stats(self):
        for provider in self.providers:
            self.provider_stats[provider.name] = {
                "requests": 0,
                "active_connections": 0,
                "total_latency": 0,
                "error_count": 0,
                "last_error": None,
                "health_status": "healthy",
                "weight": 1.0
            }

    async def get_provider(self) -> Optional[BaseProvider]:
        healthy_providers = self._get_healthy_providers()
        
        if not healthy_providers:
            logger.error("No healthy providers available")
            return None

        if self.strategy == "round_robin":
            return self._round_robin_select(healthy_providers)
        elif self.strategy == "least_connections":
            return self._least_connections_select(healthy_providers)
        elif self.strategy == "weighted":
            return self._weighted_select(healthy_providers)
        elif self.strategy == "latency_based":
            return self._latency_based_select(healthy_providers)
        else:
            return random.choice(healthy_providers)

    def _get_healthy_providers(self) -> List[BaseProvider]:
        healthy = []
        for provider in self.providers:
            stats = self.provider_stats.get(provider.name, {})
            if stats.get("health_status") == "healthy":
                healthy.append(provider)
        return healthy

    def _round_robin_select(self, providers: List[BaseProvider]) -> BaseProvider:
        provider = providers[self.current_index % len(providers)]
        self.current_index += 1
        return provider

    def _least_connections_select(self, providers: List[BaseProvider]) -> BaseProvider:
        min_connections = float('inf')
        selected = providers[0]
        
        for provider in providers:
            connections = self.provider_stats[provider.name]["active_connections"]
            if connections < min_connections:
                min_connections = connections
                selected = provider
        
        return selected

    def _weighted_select(self, providers: List[BaseProvider]) -> BaseProvider:
        weights = [self.provider_stats[p.name]["weight"] for p in providers]
        return random.choices(providers, weights=weights, k=1)[0]

    def _latency_based_select(self, providers: List[BaseProvider]) -> BaseProvider:
        best_latency = float('inf')
        selected = providers[0]
        
        for provider in providers:
            stats = self.provider_stats[provider.name]
            if stats["requests"] > 0:
                avg_latency = stats["total_latency"] / stats["requests"]
                if avg_latency < best_latency:
                    best_latency = avg_latency
                    selected = provider
        
        return selected

    async def mark_request_start(self, provider_name: str):
        if provider_name in self.provider_stats:
            self.provider_stats[provider_name]["active_connections"] += 1
            self.provider_stats[provider_name]["requests"] += 1

    async def mark_request_end(self, provider_name: str, latency: float, success: bool = True):
        if provider_name in self.provider_stats:
            stats = self.provider_stats[provider_name]
            stats["active_connections"] = max(0, stats["active_connections"] - 1)
            stats["total_latency"] += latency
            
            if not success:
                stats["error_count"] += 1
                stats["last_error"] = datetime.now()
                
                # Mark unhealthy if too many errors
                if stats["error_count"] > 5:
                    stats["health_status"] = "unhealthy"
                    asyncio.create_task(self._health_check_recovery(provider_name))

    async def _health_check_recovery(self, provider_name: str):
        await asyncio.sleep(60)  # Wait before retrying
        
        # Try to recover the provider
        for provider in self.providers:
            if provider.name == provider_name:
                try:
                    # Implement actual health check here
                    self.provider_stats[provider_name]["health_status"] = "healthy"
                    self.provider_stats[provider_name]["error_count"] = 0
                    logger.info(f"Provider {provider_name} recovered")
                except Exception as e:
                    logger.error(f"Provider {provider_name} still unhealthy: {e}")
                    asyncio.create_task(self._health_check_recovery(provider_name))

    def get_stats(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "providers": self.provider_stats,
            "healthy_count": len(self._get_healthy_providers()),
            "total_providers": len(self.providers)
        }