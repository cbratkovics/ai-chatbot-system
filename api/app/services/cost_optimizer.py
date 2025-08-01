"""
Advanced Cost Optimization System
Implements intelligent model routing, token prediction, and usage analytics
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import aioredis
from prometheus_client import Counter, Histogram, Gauge
import logging

logger = logging.getLogger(__name__)

# Metrics
cache_hits = Counter('cache_hits_total', 'Total cache hits', ['cache_type'])
cache_misses = Counter('cache_misses_total', 'Total cache misses', ['cache_type'])
tokens_saved = Counter('tokens_saved_total', 'Total tokens saved by optimization')
cost_saved = Counter('cost_saved_dollars', 'Total cost saved in dollars')
model_usage = Counter('model_usage_total', 'Model usage by type', ['model'])
optimization_latency = Histogram('optimization_latency_seconds', 'Optimization processing time')


class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class ModelTier(Enum):
    ECONOMY = "economy"      # GPT-3.5-turbo
    STANDARD = "standard"    # GPT-4-turbo
    PREMIUM = "premium"      # GPT-4
    ULTRA = "ultra"         # GPT-4-32k or Claude-3-opus


@dataclass
class ModelConfig:
    name: str
    tier: ModelTier
    input_cost: float  # $ per 1K tokens
    output_cost: float  # $ per 1K tokens
    max_tokens: int
    latency_ms: float
    quality_score: float  # 0-1
    capabilities: List[str] = field(default_factory=list)


@dataclass
class OptimizationResult:
    model: str
    estimated_tokens: int
    estimated_cost: float
    cache_hit: bool = False
    cached_response: Optional[str] = None
    compression_ratio: float = 1.0
    similarity_score: float = 0.0
    

class CostOptimizer:
    """Advanced cost optimization system with intelligent routing and caching"""
    
    # Model configurations with 2024 pricing
    MODELS = {
        "gpt-3.5-turbo": ModelConfig(
            name="gpt-3.5-turbo",
            tier=ModelTier.ECONOMY,
            input_cost=0.0005,
            output_cost=0.0015,
            max_tokens=16385,
            latency_ms=500,
            quality_score=0.7,
            capabilities=["general", "simple_qa", "basic_reasoning"]
        ),
        "gpt-4-turbo": ModelConfig(
            name="gpt-4-turbo-preview",
            tier=ModelTier.STANDARD,
            input_cost=0.01,
            output_cost=0.03,
            max_tokens=128000,
            latency_ms=1000,
            quality_score=0.85,
            capabilities=["general", "complex_reasoning", "coding", "analysis"]
        ),
        "gpt-4": ModelConfig(
            name="gpt-4",
            tier=ModelTier.PREMIUM,
            input_cost=0.03,
            output_cost=0.06,
            max_tokens=8192,
            latency_ms=2000,
            quality_score=0.95,
            capabilities=["general", "expert_reasoning", "creative", "technical"]
        ),
        "claude-3-haiku": ModelConfig(
            name="claude-3-haiku",
            tier=ModelTier.ECONOMY,
            input_cost=0.00025,
            output_cost=0.00125,
            max_tokens=200000,
            latency_ms=400,
            quality_score=0.65,
            capabilities=["general", "simple_qa", "summarization"]
        ),
        "claude-3-sonnet": ModelConfig(
            name="claude-3-sonnet",
            tier=ModelTier.STANDARD,
            input_cost=0.003,
            output_cost=0.015,
            max_tokens=200000,
            latency_ms=800,
            quality_score=0.8,
            capabilities=["general", "reasoning", "coding", "multilingual"]
        ),
        "claude-3-opus": ModelConfig(
            name="claude-3-opus",
            tier=ModelTier.ULTRA,
            input_cost=0.015,
            output_cost=0.075,
            max_tokens=200000,
            latency_ms=1500,
            quality_score=0.98,
            capabilities=["general", "expert_reasoning", "research", "complex_analysis"]
        )
    }
    
    def __init__(self, redis_client: aioredis.Redis, embedding_model_name: str = "text-embedding-ada-002"):
        self.redis = redis_client
        self.embedding_model = embedding_model_name
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.similarity_threshold = 0.93
        self.token_encoders = {
            "gpt": tiktoken.encoding_for_model("gpt-4"),
            "claude": tiktoken.encoding_for_model("gpt-4")  # Approximate
        }
        
        # Usage tracking
        self.tenant_usage: Dict[str, Dict[str, float]] = {}
        self.budget_limits: Dict[str, float] = {}
        
    async def optimize_request(
        self,
        messages: List[Dict[str, str]],
        tenant_id: str,
        requirements: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """Optimize request with intelligent routing and caching"""
        start_time = time.time()
        
        # Check semantic cache first
        cache_result = await self._check_semantic_cache(messages)
        if cache_result:
            optimization_latency.observe(time.time() - start_time)
            return cache_result
            
        # Analyze query complexity
        complexity = self._analyze_complexity(messages)
        
        # Predict token usage
        estimated_tokens = self._predict_tokens(messages, complexity)
        
        # Check budget constraints
        if not await self._check_budget(tenant_id, estimated_tokens):
            # Force economy model if over budget
            model = self._select_model(complexity, requirements, force_economy=True)
        else:
            # Select optimal model
            model = self._select_model(complexity, requirements)
            
        # Compress messages if needed
        compressed_messages = await self._compress_messages(messages, model.max_tokens)
        
        # Calculate final cost
        final_tokens = self._count_tokens(compressed_messages)
        estimated_cost = self._calculate_cost(model, final_tokens, estimated_tokens // 2)
        
        # Update usage tracking
        await self._update_usage(tenant_id, model.name, estimated_cost)
        
        optimization_latency.observe(time.time() - start_time)
        
        return OptimizationResult(
            model=model.name,
            estimated_tokens=final_tokens,
            estimated_cost=estimated_cost,
            compression_ratio=len(str(messages)) / len(str(compressed_messages))
        )
        
    async def _check_semantic_cache(self, messages: List[Dict[str, str]]) -> Optional[OptimizationResult]:
        """Check semantic cache for similar queries"""
        query = self._messages_to_text(messages)
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Check exact match first
        exact_match = await self.redis.get(f"cache:exact:{query_hash}")
        if exact_match:
            cache_hits.labels(cache_type="exact").inc()
            cached_data = json.loads(exact_match)
            return OptimizationResult(
                model=cached_data["model"],
                estimated_tokens=0,
                estimated_cost=0,
                cache_hit=True,
                cached_response=cached_data["response"],
                similarity_score=1.0
            )
            
        # Get query embedding
        query_embedding = await self._get_embedding(query)
        
        # Search for similar queries
        similar_keys = await self.redis.keys("cache:semantic:*")
        best_match = None
        best_score = 0
        
        for key in similar_keys[:100]:  # Limit search
            cached_data = await self.redis.get(key)
            if not cached_data:
                continue
                
            cached = json.loads(cached_data)
            cached_embedding = np.array(cached["embedding"])
            
            # Calculate similarity
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                cached_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > self.similarity_threshold and similarity > best_score:
                best_score = similarity
                best_match = cached
                
        if best_match:
            cache_hits.labels(cache_type="semantic").inc()
            
            # Save tokens by using cached response
            tokens_saved.inc(best_match["tokens"])
            cost_saved.inc(best_match["cost"])
            
            return OptimizationResult(
                model=best_match["model"],
                estimated_tokens=0,
                estimated_cost=0,
                cache_hit=True,
                cached_response=best_match["response"],
                similarity_score=best_score
            )
            
        cache_misses.labels(cache_type="semantic").inc()
        return None
        
    def _analyze_complexity(self, messages: List[Dict[str, str]]) -> QueryComplexity:
        """Analyze query complexity using multiple signals"""
        text = self._messages_to_text(messages)
        
        # Signal 1: Message length
        length_score = min(len(text) / 1000, 1.0)  # Normalize to 0-1
        
        # Signal 2: Number of turns
        turn_score = min(len(messages) / 10, 1.0)
        
        # Signal 3: Keyword indicators
        complex_keywords = [
            "analyze", "compare", "evaluate", "explain why", "how does",
            "implement", "design", "architecture", "optimize", "debug"
        ]
        keyword_score = sum(1 for kw in complex_keywords if kw in text.lower()) / len(complex_keywords)
        
        # Signal 4: Code detection
        code_indicators = ["```", "def ", "class ", "function ", "import ", "return "]
        code_score = 1.0 if any(ind in text for ind in code_indicators) else 0.0
        
        # Signal 5: Question complexity (question marks, subordinate clauses)
        question_score = text.count("?") / 10
        
        # Weighted combination
        complexity_score = (
            length_score * 0.2 +
            turn_score * 0.2 +
            keyword_score * 0.3 +
            code_score * 0.2 +
            question_score * 0.1
        )
        
        # Map to complexity levels
        if complexity_score < 0.25:
            return QueryComplexity.SIMPLE
        elif complexity_score < 0.5:
            return QueryComplexity.MODERATE
        elif complexity_score < 0.75:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.EXPERT
            
    def _predict_tokens(self, messages: List[Dict[str, str]], complexity: QueryComplexity) -> int:
        """Predict token usage based on input and complexity"""
        input_tokens = self._count_tokens(messages)
        
        # Output multipliers based on complexity
        multipliers = {
            QueryComplexity.SIMPLE: 1.5,
            QueryComplexity.MODERATE: 2.5,
            QueryComplexity.COMPLEX: 4.0,
            QueryComplexity.EXPERT: 6.0
        }
        
        predicted_output = int(input_tokens * multipliers[complexity])
        
        # Add buffer for safety
        return input_tokens + predicted_output + 100
        
    def _select_model(
        self,
        complexity: QueryComplexity,
        requirements: Optional[Dict[str, Any]] = None,
        force_economy: bool = False
    ) -> ModelConfig:
        """Select optimal model based on complexity and requirements"""
        if force_economy:
            model_usage.labels(model="gpt-3.5-turbo").inc()
            return self.MODELS["gpt-3.5-turbo"]
            
        # Check explicit requirements
        if requirements:
            if "model" in requirements:
                model_name = requirements["model"]
                if model_name in self.MODELS:
                    model_usage.labels(model=model_name).inc()
                    return self.MODELS[model_name]
                    
            if "max_latency_ms" in requirements:
                # Filter by latency requirement
                valid_models = [
                    m for m in self.MODELS.values()
                    if m.latency_ms <= requirements["max_latency_ms"]
                ]
                if not valid_models:
                    valid_models = list(self.MODELS.values())
                    
        # Default mapping based on complexity
        complexity_mapping = {
            QueryComplexity.SIMPLE: ["gpt-3.5-turbo", "claude-3-haiku"],
            QueryComplexity.MODERATE: ["gpt-4-turbo", "claude-3-sonnet"],
            QueryComplexity.COMPLEX: ["gpt-4", "claude-3-sonnet"],
            QueryComplexity.EXPERT: ["gpt-4", "claude-3-opus"]
        }
        
        preferred_models = complexity_mapping[complexity]
        
        # Select based on availability and cost
        for model_name in preferred_models:
            if model_name in self.MODELS:
                model_usage.labels(model=model_name).inc()
                return self.MODELS[model_name]
                
        # Fallback
        model_usage.labels(model="gpt-4-turbo").inc()
        return self.MODELS["gpt-4-turbo"]
        
    async def _compress_messages(self, messages: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
        """Compress messages to fit within token limits"""
        current_tokens = self._count_tokens(messages)
        
        if current_tokens <= max_tokens * 0.8:  # Leave 20% buffer
            return messages
            
        compressed = []
        
        for msg in messages:
            if msg["role"] == "system":
                # Keep system messages intact
                compressed.append(msg)
            else:
                # Compress user/assistant messages
                compressed_content = await self._compress_text(msg["content"])
                compressed.append({
                    "role": msg["role"],
                    "content": compressed_content
                })
                
        # If still too long, truncate older messages
        while self._count_tokens(compressed) > max_tokens * 0.8 and len(compressed) > 2:
            # Keep system and last message
            compressed.pop(1)
            
        return compressed
        
    async def _compress_text(self, text: str) -> str:
        """Compress text while preserving meaning"""
        # Remove redundant whitespace
        text = " ".join(text.split())
        
        # Remove filler phrases
        filler_phrases = [
            "I think that", "In my opinion", "It seems like",
            "As you know", "Obviously", "Basically"
        ]
        
        for phrase in filler_phrases:
            text = text.replace(phrase, "")
            
        # Summarize long paragraphs
        if len(text) > 1000:
            # In production, use a summarization model
            sentences = text.split(". ")
            if len(sentences) > 5:
                # Keep first 2 and last 2 sentences
                text = ". ".join(sentences[:2] + ["..."] + sentences[-2:])
                
        return text.strip()
        
    async def _check_budget(self, tenant_id: str, estimated_tokens: int) -> bool:
        """Check if request fits within tenant budget"""
        if tenant_id not in self.budget_limits:
            return True  # No limit set
            
        current_usage = await self._get_current_usage(tenant_id)
        estimated_cost = estimated_tokens * 0.00003  # Average cost estimate
        
        return current_usage + estimated_cost <= self.budget_limits[tenant_id]
        
    async def _update_usage(self, tenant_id: str, model: str, cost: float):
        """Update usage tracking for tenant"""
        key = f"usage:{tenant_id}:{time.strftime('%Y-%m')}"
        
        # Update total usage
        await self.redis.hincrbyfloat(key, "total_cost", cost)
        await self.redis.hincrbyfloat(key, f"model:{model}", cost)
        
        # Set expiry (keep for 6 months)
        await self.redis.expire(key, 86400 * 180)
        
        # Check for budget alerts
        if tenant_id in self.budget_limits:
            current_usage = await self._get_current_usage(tenant_id)
            limit = self.budget_limits[tenant_id]
            
            if current_usage > limit * 0.8 and current_usage - cost <= limit * 0.8:
                # Send 80% budget alert
                await self._send_budget_alert(tenant_id, current_usage, limit, 80)
            elif current_usage > limit * 0.95 and current_usage - cost <= limit * 0.95:
                # Send 95% budget alert
                await self._send_budget_alert(tenant_id, current_usage, limit, 95)
                
    async def _get_current_usage(self, tenant_id: str) -> float:
        """Get current month usage for tenant"""
        key = f"usage:{tenant_id}:{time.strftime('%Y-%m')}"
        usage = await self.redis.hget(key, "total_cost")
        return float(usage) if usage else 0.0
        
    async def _send_budget_alert(self, tenant_id: str, current: float, limit: float, percentage: int):
        """Send budget alert to tenant"""
        alert = {
            "tenant_id": tenant_id,
            "current_usage": current,
            "budget_limit": limit,
            "percentage": percentage,
            "timestamp": time.time()
        }
        
        # Publish to alert channel
        await self.redis.publish("budget_alerts", json.dumps(alert))
        logger.warning(f"Budget alert for {tenant_id}: {percentage}% of ${limit:.2f} used")
        
    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in messages"""
        encoder = self.token_encoders["gpt"]
        total = 0
        
        for msg in messages:
            total += len(encoder.encode(msg.get("content", "")))
            total += 4  # Message overhead
            
        return total
        
    def _calculate_cost(self, model: ModelConfig, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given model and tokens"""
        input_cost = (input_tokens / 1000) * model.input_cost
        output_cost = (output_tokens / 1000) * model.output_cost
        return input_cost + output_cost
        
    def _messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to single text for analysis"""
        return " ".join(msg.get("content", "") for msg in messages)
        
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get text embedding (cached)"""
        # In production, use actual embedding API
        # Here we use TF-IDF as placeholder
        if text not in self.embedding_cache:
            # Simulate embedding
            words = text.lower().split()[:100]  # Limit length
            embedding = np.random.rand(768)  # Simulate 768-dim embedding
            self.embedding_cache[text] = embedding
            
        return self.embedding_cache[text]
        
    async def cache_response(
        self,
        messages: List[Dict[str, str]],
        response: str,
        model: str,
        tokens_used: int,
        cost: float
    ):
        """Cache response for future use"""
        query = self._messages_to_text(messages)
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Cache exact match
        exact_data = {
            "query": query,
            "response": response,
            "model": model,
            "tokens": tokens_used,
            "cost": cost,
            "timestamp": time.time()
        }
        
        await self.redis.setex(
            f"cache:exact:{query_hash}",
            86400,  # 24 hour TTL
            json.dumps(exact_data)
        )
        
        # Cache with embedding for semantic search
        embedding = await self._get_embedding(query)
        semantic_data = exact_data.copy()
        semantic_data["embedding"] = embedding.tolist()
        
        await self.redis.setex(
            f"cache:semantic:{query_hash}",
            86400 * 7,  # 7 day TTL for semantic cache
            json.dumps(semantic_data)
        )
        
    def set_budget_limit(self, tenant_id: str, monthly_limit: float):
        """Set monthly budget limit for tenant"""
        self.budget_limits[tenant_id] = monthly_limit
        
    async def get_usage_analytics(self, tenant_id: str) -> Dict[str, Any]:
        """Get detailed usage analytics for tenant"""
        current_month = time.strftime('%Y-%m')
        key = f"usage:{tenant_id}:{current_month}"
        
        usage_data = await self.redis.hgetall(key)
        
        analytics = {
            "tenant_id": tenant_id,
            "period": current_month,
            "total_cost": float(usage_data.get("total_cost", 0)),
            "model_breakdown": {},
            "daily_usage": [],
            "recommendations": []
        }
        
        # Model breakdown
        for field, value in usage_data.items():
            if field.startswith("model:"):
                model = field.replace("model:", "")
                analytics["model_breakdown"][model] = float(value)
                
        # Add recommendations
        if analytics["total_cost"] > 0:
            analytics["recommendations"] = self._generate_cost_recommendations(analytics)
            
        return analytics
        
    def _generate_cost_recommendations(self, analytics: Dict[str, Any]) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        # Check model usage patterns
        model_breakdown = analytics["model_breakdown"]
        total_cost = analytics["total_cost"]
        
        if "gpt-4" in model_breakdown:
            gpt4_percentage = model_breakdown["gpt-4"] / total_cost
            if gpt4_percentage > 0.5:
                recommendations.append(
                    "Consider using GPT-4-Turbo for 30% cost reduction on complex queries"
                )
                
        # Check for optimization opportunities
        if total_cost > 1000:
            recommendations.append(
                "Enable semantic caching to reduce costs by up to 40% on repeated queries"
            )
            
        return recommendations