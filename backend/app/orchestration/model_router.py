"""Intelligent model routing and selection logic."""

import asyncio
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
from datetime import datetime, timedelta

from ..providers.base_provider import (
    BaseProvider,
    CompletionResponse,
    Message,
    ModelConfig,
    ProviderStatus
)
from ..providers.openai_adapter import OpenAIAdapter
from ..providers.anthropic_adapter import AnthropicAdapter
from ..providers.llama_adapter import LlamaAdapter


class RoutingStrategy(Enum):
    """Model routing strategies."""
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    BALANCED = "balanced"
    ROUND_ROBIN = "round_robin"
    LEAST_LATENCY = "least_latency"
    FAILOVER = "failover"
    CAPABILITY_BASED = "capability_based"


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class ModelScore:
    """Score for model selection."""
    provider: str
    model: str
    cost_score: float
    performance_score: float
    availability_score: float
    total_score: float
    estimated_cost: float
    estimated_latency_ms: float


@dataclass
class RoutingDecision:
    """Routing decision result."""
    provider: BaseProvider
    model_config: ModelConfig
    fallback_options: List[Tuple[BaseProvider, ModelConfig]]
    strategy_used: RoutingStrategy
    complexity: QueryComplexity
    reasoning: str


class ModelRouter:
    """Intelligent model router for optimal selection and failover."""
    
    # Model capability mapping
    MODEL_CAPABILITIES = {
        "gpt-4-turbo-preview": {"max_complexity": QueryComplexity.EXPERT, "cost": 0.04, "latency": 2000},
        "gpt-4": {"max_complexity": QueryComplexity.EXPERT, "cost": 0.09, "latency": 3000},
        "gpt-3.5-turbo": {"max_complexity": QueryComplexity.MODERATE, "cost": 0.002, "latency": 1000},
        "claude-3-opus": {"max_complexity": QueryComplexity.EXPERT, "cost": 0.09, "latency": 2500},
        "claude-3-sonnet": {"max_complexity": QueryComplexity.COMPLEX, "cost": 0.018, "latency": 1500},
        "claude-3-haiku": {"max_complexity": QueryComplexity.MODERATE, "cost": 0.00138, "latency": 800},
        "llama3": {"max_complexity": QueryComplexity.MODERATE, "cost": 0.0002, "latency": 500},
        "llama3:70b": {"max_complexity": QueryComplexity.COMPLEX, "cost": 0.0006, "latency": 1200},
    }
    
    def __init__(
        self,
        providers: Dict[str, BaseProvider],
        default_strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        cache_enabled: bool = True
    ):
        """
        Initialize model router.
        
        Args:
            providers: Dictionary of provider name to provider instance
            default_strategy: Default routing strategy
            cache_enabled: Whether to use cache for routing decisions
        """
        self.providers = providers
        self.default_strategy = default_strategy
        self.cache_enabled = cache_enabled
        self._routing_cache: Dict[str, RoutingDecision] = {}
        self._provider_stats: Dict[str, Dict[str, Any]] = {}
        self._round_robin_index = 0
    
    def analyze_query_complexity(self, messages: List[Message]) -> QueryComplexity:
        """Analyze query complexity based on messages."""
        # Combine all message content
        full_text = " ".join([msg.content for msg in messages])
        word_count = len(full_text.split())
        
        # Check for complexity indicators
        complex_indicators = [
            "explain", "analyze", "compare", "evaluate", "design",
            "implement", "architecture", "algorithm", "optimize",
            "debug", "troubleshoot", "refactor"
        ]
        
        expert_indicators = [
            "quantum", "distributed", "concurrent", "parallel",
            "machine learning", "neural network", "blockchain",
            "cryptography", "compiler", "operating system"
        ]
        
        indicator_count = sum(1 for word in complex_indicators if word in full_text.lower())
        expert_count = sum(1 for word in expert_indicators if word in full_text.lower())
        
        # Determine complexity
        if expert_count > 0 or word_count > 500:
            return QueryComplexity.EXPERT
        elif indicator_count >= 3 or word_count > 200:
            return QueryComplexity.COMPLEX
        elif indicator_count >= 1 or word_count > 50:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _get_cache_key(self, messages: List[Message], strategy: RoutingStrategy) -> str:
        """Generate cache key for routing decision."""
        # Create a hash of the messages and strategy
        content = json.dumps([msg.content for msg in messages[-3:]])  # Last 3 messages
        key = f"{strategy.value}:{hashlib.md5(content.encode()).hexdigest()}"
        return key
    
    def _score_models(
        self,
        complexity: QueryComplexity,
        strategy: RoutingStrategy,
        max_tokens: int
    ) -> List[ModelScore]:
        """Score available models based on strategy and complexity."""
        scores = []
        
        for provider_name, provider in self.providers.items():
            if not provider.is_available():
                continue
            
            for model in provider.supported_models:
                if model not in self.MODEL_CAPABILITIES:
                    continue
                
                capabilities = self.MODEL_CAPABILITIES[model]
                
                # Check if model can handle complexity
                if self._compare_complexity(complexity, capabilities["max_complexity"]) > 0:
                    continue
                
                # Calculate scores
                cost_score = 1.0 / (1.0 + capabilities["cost"])  # Lower cost = higher score
                performance_score = 1.0 / (1.0 + capabilities["latency"] / 1000)  # Lower latency = higher score
                
                # Get provider availability
                provider_metrics = provider.get_metrics()
                availability_score = 1.0 - provider_metrics.error_rate
                
                # Calculate total score based on strategy
                if strategy == RoutingStrategy.COST_OPTIMIZED:
                    total_score = cost_score * 0.7 + performance_score * 0.1 + availability_score * 0.2
                elif strategy == RoutingStrategy.PERFORMANCE_OPTIMIZED:
                    total_score = cost_score * 0.1 + performance_score * 0.7 + availability_score * 0.2
                elif strategy == RoutingStrategy.BALANCED:
                    total_score = cost_score * 0.33 + performance_score * 0.33 + availability_score * 0.34
                else:
                    total_score = (cost_score + performance_score + availability_score) / 3
                
                # Estimate actual cost
                estimated_tokens = max_tokens * 2  # Rough estimate including prompt
                estimated_cost = capabilities["cost"] * (estimated_tokens / 1000)
                
                scores.append(ModelScore(
                    provider=provider_name,
                    model=model,
                    cost_score=cost_score,
                    performance_score=performance_score,
                    availability_score=availability_score,
                    total_score=total_score,
                    estimated_cost=estimated_cost,
                    estimated_latency_ms=capabilities["latency"]
                ))
        
        # Sort by total score (descending)
        scores.sort(key=lambda x: x.total_score, reverse=True)
        return scores
    
    def _compare_complexity(self, c1: QueryComplexity, c2: QueryComplexity) -> int:
        """Compare two complexity levels. Returns -1 if c1 < c2, 0 if equal, 1 if c1 > c2."""
        complexity_order = {
            QueryComplexity.SIMPLE: 0,
            QueryComplexity.MODERATE: 1,
            QueryComplexity.COMPLEX: 2,
            QueryComplexity.EXPERT: 3
        }
        return complexity_order[c1] - complexity_order[c2]
    
    async def route(
        self,
        messages: List[Message],
        model_config: ModelConfig,
        strategy: Optional[RoutingStrategy] = None,
        tenant_preferences: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Route request to optimal model.
        
        Args:
            messages: Chat messages
            model_config: Model configuration
            strategy: Routing strategy override
            tenant_preferences: Tenant-specific preferences
        
        Returns:
            RoutingDecision with selected provider and fallbacks
        """
        strategy = strategy or self.default_strategy
        
        # Check cache
        if self.cache_enabled:
            cache_key = self._get_cache_key(messages, strategy)
            if cache_key in self._routing_cache:
                cached_decision = self._routing_cache[cache_key]
                # Validate provider is still available
                if cached_decision.provider.is_available():
                    return cached_decision
        
        # Analyze query complexity
        complexity = self.analyze_query_complexity(messages)
        
        # Apply tenant preferences if provided
        if tenant_preferences:
            if "preferred_model" in tenant_preferences:
                model_config.model_id = tenant_preferences["preferred_model"]
            if "max_cost" in tenant_preferences:
                # Filter models by cost constraint
                pass
        
        # Route based on strategy
        if strategy == RoutingStrategy.ROUND_ROBIN:
            decision = await self._route_round_robin(model_config, complexity)
        elif strategy == RoutingStrategy.LEAST_LATENCY:
            decision = await self._route_least_latency(model_config, complexity)
        elif strategy == RoutingStrategy.FAILOVER:
            decision = await self._route_failover(model_config, complexity)
        else:
            # Score-based routing
            scores = self._score_models(complexity, strategy, model_config.max_tokens)
            
            if not scores:
                raise Exception("No available models for routing")
            
            # Select primary model
            primary = scores[0]
            provider = self.providers[primary.provider]
            
            # Update model in config
            config = ModelConfig(**model_config.__dict__)
            config.model_id = primary.model
            
            # Select fallback options
            fallbacks = []
            for score in scores[1:4]:  # Top 3 fallbacks
                fallback_provider = self.providers[score.provider]
                fallback_config = ModelConfig(**model_config.__dict__)
                fallback_config.model_id = score.model
                fallbacks.append((fallback_provider, fallback_config))
            
            reasoning = (
                f"Selected {primary.model} from {primary.provider} "
                f"(score: {primary.total_score:.3f}, cost: ${primary.estimated_cost:.4f}, "
                f"latency: {primary.estimated_latency_ms}ms) "
                f"for {complexity.value} complexity query using {strategy.value} strategy"
            )
            
            decision = RoutingDecision(
                provider=provider,
                model_config=config,
                fallback_options=fallbacks,
                strategy_used=strategy,
                complexity=complexity,
                reasoning=reasoning
            )
        
        # Cache decision
        if self.cache_enabled:
            self._routing_cache[cache_key] = decision
            # Clean old cache entries
            if len(self._routing_cache) > 1000:
                self._routing_cache = dict(list(self._routing_cache.items())[-500:])
        
        return decision
    
    async def _route_round_robin(
        self,
        model_config: ModelConfig,
        complexity: QueryComplexity
    ) -> RoutingDecision:
        """Round-robin routing across available providers."""
        available_providers = [
            (name, provider) for name, provider in self.providers.items()
            if provider.is_available()
        ]
        
        if not available_providers:
            raise Exception("No available providers for routing")
        
        # Select next provider in rotation
        self._round_robin_index = (self._round_robin_index + 1) % len(available_providers)
        selected_name, selected_provider = available_providers[self._round_robin_index]
        
        # Select appropriate model for complexity
        suitable_models = [
            model for model in selected_provider.supported_models
            if model in self.MODEL_CAPABILITIES
            and self._compare_complexity(complexity, self.MODEL_CAPABILITIES[model]["max_complexity"]) <= 0
        ]
        
        if not suitable_models:
            suitable_models = selected_provider.supported_models
        
        config = ModelConfig(**model_config.__dict__)
        config.model_id = suitable_models[0]
        
        # Set other providers as fallbacks
        fallbacks = []
        for name, provider in available_providers:
            if name != selected_name and provider.supported_models:
                fallback_config = ModelConfig(**model_config.__dict__)
                fallback_config.model_id = provider.supported_models[0]
                fallbacks.append((provider, fallback_config))
        
        return RoutingDecision(
            provider=selected_provider,
            model_config=config,
            fallback_options=fallbacks,
            strategy_used=RoutingStrategy.ROUND_ROBIN,
            complexity=complexity,
            reasoning=f"Round-robin selection: {selected_name}"
        )
    
    async def _route_least_latency(
        self,
        model_config: ModelConfig,
        complexity: QueryComplexity
    ) -> RoutingDecision:
        """Route to provider with least latency."""
        best_option = None
        best_latency = float('inf')
        
        for provider_name, provider in self.providers.items():
            if not provider.is_available():
                continue
            
            metrics = provider.get_metrics()
            if metrics.average_latency_ms < best_latency:
                best_latency = metrics.average_latency_ms
                best_option = (provider_name, provider)
        
        if not best_option:
            raise Exception("No available providers for routing")
        
        selected_name, selected_provider = best_option
        
        # Select fastest model
        fastest_model = None
        min_latency = float('inf')
        
        for model in selected_provider.supported_models:
            if model in self.MODEL_CAPABILITIES:
                latency = self.MODEL_CAPABILITIES[model]["latency"]
                if latency < min_latency:
                    min_latency = latency
                    fastest_model = model
        
        config = ModelConfig(**model_config.__dict__)
        config.model_id = fastest_model or selected_provider.supported_models[0]
        
        return RoutingDecision(
            provider=selected_provider,
            model_config=config,
            fallback_options=[],
            strategy_used=RoutingStrategy.LEAST_LATENCY,
            complexity=complexity,
            reasoning=f"Least latency: {selected_name} ({best_latency:.1f}ms avg)"
        )
    
    async def _route_failover(
        self,
        model_config: ModelConfig,
        complexity: QueryComplexity
    ) -> RoutingDecision:
        """Failover routing with priority order."""
        # Priority order for failover
        priority_order = ["OpenAI", "Anthropic", "Llama"]
        
        for provider_name in priority_order:
            if provider_name not in self.providers:
                continue
            
            provider = self.providers[provider_name]
            if provider.is_available():
                # Select best model for complexity
                suitable_models = [
                    model for model in provider.supported_models
                    if model in self.MODEL_CAPABILITIES
                ]
                
                if suitable_models:
                    config = ModelConfig(**model_config.__dict__)
                    config.model_id = suitable_models[0]
                    
                    # Add other providers as fallbacks
                    fallbacks = []
                    for other_name in priority_order:
                        if other_name != provider_name and other_name in self.providers:
                            other_provider = self.providers[other_name]
                            if other_provider.supported_models:
                                fallback_config = ModelConfig(**model_config.__dict__)
                                fallback_config.model_id = other_provider.supported_models[0]
                                fallbacks.append((other_provider, fallback_config))
                    
                    return RoutingDecision(
                        provider=provider,
                        model_config=config,
                        fallback_options=fallbacks,
                        strategy_used=RoutingStrategy.FAILOVER,
                        complexity=complexity,
                        reasoning=f"Failover selection: {provider_name} (priority: {priority_order.index(provider_name) + 1})"
                    )
        
        raise Exception("No available providers for failover routing")
    
    async def execute_with_fallback(
        self,
        messages: List[Message],
        routing_decision: RoutingDecision
    ) -> CompletionResponse:
        """Execute request with automatic fallback on failure."""
        # Try primary provider
        try:
            response = await routing_decision.provider.complete(
                messages,
                routing_decision.model_config
            )
            return response
        except Exception as primary_error:
            # Log primary failure
            print(f"Primary provider failed: {primary_error}")
            
            # Try fallback options
            for fallback_provider, fallback_config in routing_decision.fallback_options:
                try:
                    response = await fallback_provider.complete(
                        messages,
                        fallback_config
                    )
                    response.metadata = response.metadata or {}
                    response.metadata["fallback_used"] = True
                    response.metadata["fallback_provider"] = fallback_provider.name
                    return response
                except Exception as fallback_error:
                    print(f"Fallback provider {fallback_provider.name} failed: {fallback_error}")
                    continue
            
            # All providers failed
            raise Exception(f"All providers failed. Primary error: {primary_error}")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        stats = {
            "cache_size": len(self._routing_cache),
            "providers": {}
        }
        
        for name, provider in self.providers.items():
            metrics = provider.get_metrics()
            stats["providers"][name] = {
                "status": provider._status.value,
                "total_requests": metrics.total_requests,
                "error_rate": metrics.error_rate,
                "average_latency_ms": metrics.average_latency_ms,
                "total_cost": metrics.total_cost
            }
        
        return stats