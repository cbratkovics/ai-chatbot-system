"""Anthropic Claude provider adapter implementation."""

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


class AnthropicAdapter(BaseProvider):
    """Anthropic API adapter for Claude models."""

    BASE_URL = "https://api.anthropic.com/v1"

    # Model pricing per 1K tokens (as of 2024)
    PRICING = {
        "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
        "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
        "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
        "claude-2.1": {"prompt": 0.008, "completion": 0.024},
        "claude-2": {"prompt": 0.008, "completion": 0.024},
        "claude-instant": {"prompt": 0.00163, "completion": 0.00551},
    }

    def __init__(self, api_key: str, config: Optional[Dict[str, Any]] = None):
        """Initialize Anthropic adapter."""
        super().__init__(api_key, config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.anthropic_version = config.get("anthropic_version", "2023-06-01") if config else "2023-06-01"

    @property
    def name(self) -> str:
        """Return provider name."""
        return "Anthropic"

    @property
    def supported_models(self) -> List[str]:
        """Return list of supported Claude models."""
        return [
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
            "claude-2.1",
            "claude-2",
            "claude-instant",
        ]

    @property
    def capabilities(self) -> List[ModelCapability]:
        """Return Anthropic capabilities."""
        return [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT_COMPLETION,
            ModelCapability.STREAMING,
            ModelCapability.VISION,  # Claude 3 models
        ]

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if not self.session:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": self.anthropic_version,
                "Content-Type": "application/json",
            }
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session

    def _convert_messages(self, messages: List[Message]) -> tuple[str, List[Dict[str, str]]]:
        """Convert internal message format to Anthropic format."""
        system_prompt = ""
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                # Anthropic uses 'user' and 'assistant' roles
                role = msg.role if msg.role in ["user", "assistant"] else "user"
                anthropic_messages.append({"role": role, "content": msg.content})

        return system_prompt, anthropic_messages

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def complete(self, messages: List[Message], model_config: ModelConfig) -> CompletionResponse:
        """Generate completion using Anthropic API."""
        if self._circuit_breaker_open:
            raise Exception(f"Circuit breaker open for {self.name}")

        session = await self._ensure_session()
        start_time = time.time()

        system_prompt, anthropic_messages = self._convert_messages(messages)

        payload = {
            "model": model_config.model_id,
            "messages": anthropic_messages,
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if model_config.stop_sequences:
            payload["stop_sequences"] = model_config.stop_sequences

        try:
            async with session.post(
                f"{self.BASE_URL}/messages", json=payload, timeout=aiohttp.ClientTimeout(total=model_config.timeout)
            ) as response:
                response.raise_for_status()
                data = await response.json()

                latency_ms = (time.time() - start_time) * 1000

                # Calculate token usage
                usage = {
                    "tokens_prompt": data.get("usage", {}).get("input_tokens", 0),
                    "tokens_completion": data.get("usage", {}).get("output_tokens", 0),
                    "tokens_total": 0,
                }
                usage["tokens_total"] = usage["tokens_prompt"] + usage["tokens_completion"]

                cost = self.estimate_cost(usage["tokens_prompt"], usage["tokens_completion"], model_config.model_id)

                self.update_metrics(success=True, tokens=usage["tokens_total"], cost=cost, latency_ms=latency_ms)

                # Extract content from response
                content = ""
                for block in data.get("content", []):
                    if block["type"] == "text":
                        content += block["text"]

                return CompletionResponse(
                    content=content,
                    model=data["model"],
                    usage=usage,
                    finish_reason=data.get("stop_reason", "stop"),
                    latency_ms=latency_ms,
                    provider=self.name,
                    metadata={"id": data["id"], "type": data["type"], "role": data["role"]},
                )

        except aiohttp.ClientResponseError as e:
            self._consecutive_failures += 1
            self.update_metrics(success=False, tokens=0, cost=0, latency_ms=0)

            if e.status == 429:  # Rate limit
                raise Exception(f"Rate limit exceeded for {self.name}")
            elif e.status == 401:
                raise Exception(f"Authentication failed for {self.name}")
            else:
                raise Exception(f"API error from {self.name}: {e.status}")

        except Exception as e:
            self._consecutive_failures += 1
            self.update_metrics(success=False, tokens=0, cost=0, latency_ms=0)
            raise Exception(f"Error calling {self.name}: {str(e)}")

    async def stream_complete(self, messages: List[Message], model_config: ModelConfig) -> AsyncIterator[StreamChunk]:
        """Stream completion from Anthropic API."""
        if self._circuit_breaker_open:
            raise Exception(f"Circuit breaker open for {self.name}")

        session = await self._ensure_session()

        system_prompt, anthropic_messages = self._convert_messages(messages)

        payload = {
            "model": model_config.model_id,
            "messages": anthropic_messages,
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
            "stream": True,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if model_config.stop_sequences:
            payload["stop_sequences"] = model_config.stop_sequences

        try:
            async with session.post(
                f"{self.BASE_URL}/messages", json=payload, timeout=aiohttp.ClientTimeout(total=model_config.timeout)
            ) as response:
                response.raise_for_status()

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data_str = line[6:]

                        try:
                            data = json.loads(data_str)

                            if data["type"] == "content_block_delta":
                                if data["delta"]["type"] == "text_delta":
                                    yield StreamChunk(delta=data["delta"]["text"], metadata={"index": data["index"]})

                            elif data["type"] == "message_stop":
                                # Final message with usage info
                                if "usage" in data:
                                    yield StreamChunk(
                                        delta="",
                                        finish_reason="stop",
                                        usage={
                                            "tokens_prompt": data["usage"]["input_tokens"],
                                            "tokens_completion": data["usage"]["output_tokens"],
                                            "tokens_total": data["usage"]["input_tokens"]
                                            + data["usage"]["output_tokens"],
                                        },
                                    )
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            self._consecutive_failures += 1
            raise Exception(f"Streaming error from {self.name}: {str(e)}")

    async def generate_embeddings(self, texts: List[str], model: str = "claude-3-haiku") -> List[List[float]]:
        """
        Generate embeddings using Anthropic.
        Note: Anthropic doesn't have dedicated embedding models,
        so this would need to use a workaround or external service.
        """
        raise NotImplementedError(
            "Anthropic does not provide native embedding models. "
            "Consider using OpenAI or another provider for embeddings."
        )

    def estimate_cost(self, tokens_prompt: int, tokens_completion: int, model: str) -> float:
        """Estimate cost for Anthropic token usage."""
        if model not in self.PRICING:
            # Default to claude-3-sonnet pricing if model not found
            model = "claude-3-sonnet"

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
