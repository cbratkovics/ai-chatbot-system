import anthropic
from typing import List, AsyncIterator, Dict
from app.services.llm.base import LLMProvider
from app.models.chat import Message
from app.config import settings
import time

class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.client = anthropic.AsyncClient(api_key=settings.anthropic_api_key)
        
        self.pricing = {
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015}
        }
    
    async def generate_response(
        self,
        messages: List[Message],
        model: str = "claude-3-opus",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> Dict:
        start_time = time.time()
        
        # Convert messages to Anthropic format
        system_message = next((m.content for m in messages if m.role == "system"), None)
        conversation = [
            {"role": "user" if m.role == "user" else "assistant", "content": m.content}
            for m in messages if m.role != "system"
        ]
        
        try:
            response = await self.client.messages.create(
                model=model,
                messages=conversation,
                system=system_message,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if not stream:
                content = response.content[0].text
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                
                return {
                    "content": content,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "cost": self.calculate_cost(input_tokens, output_tokens, model),
                    "latency_ms": (time.time() - start_time) * 1000,
                    "model": model
                }
            else:
                return response
                
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    async def stream_response(
        self,
        messages: List[Message],
        model: str = "claude-3-opus",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> AsyncIterator[str]:
        # Convert messages to Anthropic format
        system_message = next((m.content for m in messages if m.role == "system"), None)
        conversation = [
            {"role": "user" if m.role == "user" else "assistant", "content": m.content}
            for m in messages if m.role != "system"
        ]
        
        stream = await self.client.messages.create(
            model=model,
            messages=conversation,
            system=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.type == "content_block_delta":
                yield chunk.delta.text
    
    def estimate_tokens(self, text: str) -> int:
        # Rough estimation: 1 token ~= 4 characters
        return len(text) // 4
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        if model not in self.pricing:
            return 0.0
        
        input_cost = (input_tokens / 1000) * self.pricing[model]["input"]
        output_cost = (output_tokens / 1000) * self.pricing[model]["output"]
        
        return round(input_cost + output_cost, 6)