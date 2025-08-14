from typing import List, Dict, Optional
import asyncio
from app.services.llm.base import LLMProvider
from app.services.llm.openai_provider import OpenAIProvider
from app.services.llm.anthropic_provider import AnthropicProvider
from app.models.chat import Message
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class LLMOrchestrator:
    def __init__(self):
        self.providers = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider() if settings.anthropic_api_key else None
        }
        
        self.model_mapping = {
            "gpt-4": "openai",
            "gpt-3.5-turbo": "openai",
            "claude-3-opus": "anthropic",
            "claude-3-sonnet": "anthropic"
        }
        
        self.failover_chain = [
            ("gpt-4", "openai"),
            ("claude-3-opus", "anthropic"),
            ("gpt-3.5-turbo", "openai")
        ]
    
    async def generate_response_with_failover(
        self,
        messages: List[Message],
        preferred_model: str = "gpt-4",
        **kwargs
    ) -> Dict:
        """Generate response with automatic failover"""
        
        # Try preferred model first
        if preferred_model in self.model_mapping:
            provider_name = self.model_mapping[preferred_model]
            if provider_name in self.providers and self.providers[provider_name]:
                try:
                    logger.info(f"Attempting with {preferred_model}")
                    response = await self.providers[provider_name].generate_response(
                        messages=messages,
                        model=preferred_model,
                        **kwargs
                    )
                    response["provider"] = provider_name
                    return response
                except Exception as e:
                    logger.error(f"Failed with {preferred_model}: {str(e)}")
        
        # Failover chain
        for model, provider_name in self.failover_chain:
            if model == preferred_model:  # Skip already tried
                continue
                
            if provider_name in self.providers and self.providers[provider_name]:
                try:
                    logger.info(f"Failing over to {model}")
                    response = await self.providers[provider_name].generate_response(
                        messages=messages,
                        model=model,
                        **kwargs
                    )
                    response["provider"] = provider_name
                    response["failover"] = True
                    return response
                except Exception as e:
                    logger.error(f"Failed with {model}: {str(e)}")
                    continue
        
        raise Exception("All LLM providers failed")
    
    def classify_query_complexity(self, message: str) -> str:
        """Classify query to route to appropriate model"""
        # Simple heuristic - can be replaced with ML classifier
        word_count = len(message.split())
        
        if word_count < 20:
            return "simple"
        elif word_count < 100:
            return "moderate"
        else:
            return "complex"
    
    def select_model_for_query(self, message: str, optimize_cost: bool = True) -> str:
        """Select best model based on query complexity and cost"""
        complexity = self.classify_query_complexity(message)
        
        if optimize_cost:
            if complexity == "simple":
                return "gpt-3.5-turbo"
            elif complexity == "moderate":
                return "claude-3-sonnet" if settings.anthropic_api_key else "gpt-3.5-turbo"
            else:
                return "gpt-4"
        else:
            # Always use best model if not optimizing cost
            return "gpt-4" if "gpt-4" in self.model_mapping else "claude-3-opus"