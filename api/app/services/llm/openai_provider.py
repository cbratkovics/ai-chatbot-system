import openai
from typing import List, AsyncIterator, Dict
import tiktoken
from app.services.llm.base import LLMProvider
from app.models.chat import Message
from app.config import settings
import time

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = openai.AsyncClient(api_key=settings.openai_api_key)
        self.encoder = tiktoken.encoding_for_model("gpt-4")
        
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
        }
    
    async def generate_response(
        self,
        messages: List[Message],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> Dict:
        start_time = time.time()
        
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if not stream:
                content = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                
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
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def stream_response(
        self,
        messages: List[Message],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> AsyncIterator[str]:
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        stream = await self.client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def estimate_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        if model not in self.pricing:
            return 0.0
        
        input_cost = (input_tokens / 1000) * self.pricing[model]["input"]
        output_cost = (output_tokens / 1000) * self.pricing[model]["output"]
        
        return round(input_cost + output_cost, 6)