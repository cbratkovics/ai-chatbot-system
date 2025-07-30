import openai
from typing import List, AsyncIterator, Dict, Optional
import tiktoken
import json
from app.services.llm.base import LLMProvider
from app.models.chat import Message
from app.config import settings
from app.services.functions.base import function_registry
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
        stream: bool = False,
        use_functions: bool = True
    ) -> Dict:
        start_time = time.time()
        
        # Format messages for OpenAI API
        formatted_messages = []
        for msg in messages:
            if isinstance(msg.content, str):
                formatted_messages.append({"role": msg.role, "content": msg.content})
            else:
                # Handle multi-modal content
                formatted_content = []
                for content in msg.content:
                    if hasattr(content, 'type'):
                        if content.type == "text":
                            formatted_content.append({"type": "text", "text": content.text})
                        elif content.type == "image_url":
                            formatted_content.append({"type": "image_url", "image_url": content.image_url})
                    else:
                        # Fallback for dict content
                        formatted_content.append(content)
                formatted_messages.append({"role": msg.role, "content": formatted_content})
        
        # Prepare function calling parameters
        functions = None
        if use_functions and function_registry.get_all_schemas():
            functions = function_registry.get_all_schemas()
        
        try:
            # Create completion with or without functions
            create_params = {
                "model": model,
                "messages": formatted_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
            
            if functions:
                create_params["functions"] = functions
                create_params["function_call"] = "auto"
            
            response = await self.client.chat.completions.create(**create_params)
            
            if not stream:
                message = response.choices[0].message
                content = message.content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                
                # Check if function was called
                function_call = None
                if hasattr(message, 'function_call') and message.function_call:
                    function_call = {
                        "name": message.function_call.name,
                        "arguments": json.loads(message.function_call.arguments)
                    }
                    
                    # Execute the function
                    result = await function_registry.execute_function(
                        function_call["name"],
                        function_call["arguments"]
                    )
                    
                    # Add function result to messages and get final response
                    formatted_messages.append({
                        "role": "assistant",
                        "content": content,
                        "function_call": message.function_call.model_dump()
                    })
                    formatted_messages.append({
                        "role": "function",
                        "name": function_call["name"],
                        "content": json.dumps(result.result if not result.error else {"error": result.error})
                    })
                    
                    # Get final response with function result
                    final_response = await self.client.chat.completions.create(
                        model=model,
                        messages=formatted_messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    content = final_response.choices[0].message.content
                    output_tokens += final_response.usage.completion_tokens
                
                return {
                    "content": content,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "cost": self.calculate_cost(input_tokens, output_tokens, model),
                    "latency_ms": (time.time() - start_time) * 1000,
                    "model": model,
                    "function_call": function_call
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
        # Format messages for OpenAI API (same as generate_response)
        formatted_messages = []
        for msg in messages:
            if isinstance(msg.content, str):
                formatted_messages.append({"role": msg.role, "content": msg.content})
            else:
                # Handle multi-modal content
                formatted_content = []
                for content in msg.content:
                    if hasattr(content, 'type'):
                        if content.type == "text":
                            formatted_content.append({"type": "text", "text": content.text})
                        elif content.type == "image_url":
                            formatted_content.append({"type": "image_url", "image_url": content.image_url})
                    else:
                        # Fallback for dict content
                        formatted_content.append(content)
                formatted_messages.append({"role": msg.role, "content": formatted_content})
        
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