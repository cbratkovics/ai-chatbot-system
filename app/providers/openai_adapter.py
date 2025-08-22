"""OpenAI provider adapter implementation."""

import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_provider import (
    BaseProvider,
    CompletionResponse,
    Message,
    ModelCapability,
    ModelConfig,
    ProviderStatus,
    StreamChunk,
)


class OpenAIAdapter(BaseProvider):
    """OpenAI API adapter for GPT models."""

    BASE_URL = "https://api.openai.com/v1"

    # Model pricing per 1K tokens (as of 2024)
    PRICING = {
        "gpt-4-turbo-preview": {"prompt": 0.01, "completion": 0.03},
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
        "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
        "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
        "text-embedding-ada-002": {"prompt": 0.0001, "completion": 0},
    }

    def __init__(self, api_key: str, config: Optional[Dict[str, Any]] = None):
        """Initialize OpenAI adapter."""
        super().__init__(api_key, config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.org_id = config.get("organization_id") if config else None

    @property
    def name(self) -> str:
        """Return provider name."""
        return "OpenAI"

    @property
    def supported_models(self) -> List[str]:
        """Return list of supported OpenAI models."""
        return [
            "gpt-4-turbo-preview",
            "gpt-4",
            "gpt-4-32k",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "text-embedding-ada-002",
        ]

    @property
    def capabilities(self) -> List[ModelCapability]:
        """Return OpenAI capabilities."""
        return [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT_COMPLETION,
            ModelCapability.EMBEDDINGS,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.VISION,
            ModelCapability.STREAMING,
        ]

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if not self.session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            if self.org_id:
                headers["OpenAI-Organization"] = self.org_id

            self.session = aiohttp.ClientSession(headers=headers)
        return self.session

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert internal message format to OpenAI format."""
        openai_messages = []
        for msg in messages:
            openai_msg = {"role": msg.role, "content": msg.content}
            if msg.name:
                openai_msg["name"] = msg.name
            if msg.function_call:
                openai_msg["function_call"] = msg.function_call
            openai_messages.append(openai_msg)
        return openai_messages

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def complete(self, messages: List[Message], model_config: ModelConfig) -> CompletionResponse:
        """Generate completion using OpenAI API."""
        if self._circuit_breaker_open:
            raise Exception(f"Circuit breaker open for {self.name}")

        session = await self._ensure_session()
        start_time = time.time()

        payload = {
            "model": model_config.model_id,
            "messages": self._convert_messages(messages),
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
            "frequency_penalty": model_config.frequency_penalty,
            "presence_penalty": model_config.presence_penalty,
        }

        if model_config.stop_sequences:
            payload["stop"] = model_config.stop_sequences

        try:
            async with session.post(
                f"{self.BASE_URL}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=model_config.timeout),
            ) as response:
                response.raise_for_status()
                data = await response.json()

                latency_ms = (time.time() - start_time) * 1000

                usage = {
                    "tokens_prompt": data["usage"]["prompt_tokens"],
                    "tokens_completion": data["usage"]["completion_tokens"],
                    "tokens_total": data["usage"]["total_tokens"],
                }

                cost = self.estimate_cost(usage["tokens_prompt"], usage["tokens_completion"], model_config.model_id)

                self.update_metrics(success=True, tokens=usage["tokens_total"], cost=cost, latency_ms=latency_ms)

                return CompletionResponse(
                    content=data["choices"][0]["message"]["content"],
                    model=data["model"],
                    usage=usage,
                    finish_reason=data["choices"][0]["finish_reason"],
                    latency_ms=latency_ms,
                    provider=self.name,
                    metadata={"id": data["id"], "created": data["created"]},
                )

        except aiohttp.ClientResponseError as e:
            self._consecutive_failures += 1
            self.update_metrics(success=False, tokens=0, cost=0, latency_ms=0)

            if e.status == 429:  # Rate limit
                raise Exception(f"Rate limit exceeded for {self.name}")
            elif e.status == 401:
                raise Exception(f"Authentication failed for {self.name}")
            else:
                raise Exception(f"API error from {self.name}: {e.status} - {e.message}")

        except Exception as e:
            self._consecutive_failures += 1
            self.update_metrics(success=False, tokens=0, cost=0, latency_ms=0)
            raise Exception(f"Error calling {self.name}: {str(e)}")

    async def stream_complete(self, messages: List[Message], model_config: ModelConfig) -> AsyncIterator[StreamChunk]:
        """Stream completion from OpenAI API."""
        if self._circuit_breaker_open:
            raise Exception(f"Circuit breaker open for {self.name}")

        session = await self._ensure_session()

        payload = {
            "model": model_config.model_id,
            "messages": self._convert_messages(messages),
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
            "frequency_penalty": model_config.frequency_penalty,
            "presence_penalty": model_config.presence_penalty,
            "stream": True,
        }

        if model_config.stop_sequences:
            payload["stop"] = model_config.stop_sequences

        try:
            async with session.post(
                f"{self.BASE_URL}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=model_config.timeout),
            ) as response:
                response.raise_for_status()

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            choice = data["choices"][0]

                            if "delta" in choice and "content" in choice["delta"]:
                                yield StreamChunk(
                                    delta=choice["delta"]["content"],
                                    finish_reason=choice.get("finish_reason"),
                                    metadata={"id": data["id"]},
                                )
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            self._consecutive_failures += 1
            raise Exception(f"Streaming error from {self.name}: {str(e)}")

    async def generate_embeddings(self, texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        session = await self._ensure_session()

        payload = {
            "model": model,
            "input": texts,
        }

        try:
            async with session.post(
                f"{self.BASE_URL}/embeddings", json=payload, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                data = await response.json()

                embeddings = [item["embedding"] for item in data["data"]]

                # Update metrics
                tokens = data["usage"]["total_tokens"]
                cost = self.estimate_cost(tokens, 0, model)
                self.update_metrics(success=True, tokens=tokens, cost=cost, latency_ms=0)

                return embeddings

        except Exception as e:
            self.update_metrics(success=False, tokens=0, cost=0, latency_ms=0)
            raise Exception(f"Embedding error from {self.name}: {str(e)}")

    def estimate_cost(self, tokens_prompt: int, tokens_completion: int, model: str) -> float:
        """Estimate cost for OpenAI token usage."""
        if model not in self.PRICING:
            return 0.0

        pricing = self.PRICING[model]
        prompt_cost = (tokens_prompt / 1000) * pricing["prompt"]
        completion_cost = (tokens_completion / 1000) * pricing["completion"]

        return prompt_cost + completion_cost

    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    def __del__(self):
        """Cleanup on deletion."""
        if self.session:
            asyncio.create_task(self.close())
