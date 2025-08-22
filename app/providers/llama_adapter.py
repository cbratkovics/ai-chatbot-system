"""Llama model provider adapter implementation (via Ollama or custom endpoint)."""

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


class LlamaAdapter(BaseProvider):
    """Llama model adapter for local or hosted Llama models."""

    # Default Ollama endpoint
    DEFAULT_BASE_URL = "http://localhost:11434"

    # Approximate token costs (for self-hosted, mainly compute costs)
    PRICING = {
        "llama3": {"prompt": 0.0001, "completion": 0.0001},
        "llama3:70b": {"prompt": 0.0003, "completion": 0.0003},
        "llama2": {"prompt": 0.0001, "completion": 0.0001},
        "llama2:70b": {"prompt": 0.0003, "completion": 0.0003},
        "codellama": {"prompt": 0.0001, "completion": 0.0001},
        "mistral": {"prompt": 0.0001, "completion": 0.0001},
        "mixtral": {"prompt": 0.0002, "completion": 0.0002},
    }

    def __init__(self, api_key: str = "", config: Optional[Dict[str, Any]] = None):
        """
        Initialize Llama adapter.

        Args:
            api_key: Not used for local Ollama, but kept for interface consistency
            config: Configuration including base_url for the Llama endpoint
        """
        super().__init__(api_key or "local", config)
        self.base_url = config.get("base_url", self.DEFAULT_BASE_URL) if config else self.DEFAULT_BASE_URL
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_ollama = "ollama" in self.base_url or "11434" in self.base_url

    @property
    def name(self) -> str:
        """Return provider name."""
        return "Llama"

    @property
    def supported_models(self) -> List[str]:
        """Return list of supported Llama models."""
        return [
            "llama3",
            "llama3:70b",
            "llama2",
            "llama2:70b",
            "codellama",
            "mistral",
            "mixtral",
        ]

    @property
    def capabilities(self) -> List[ModelCapability]:
        """Return Llama capabilities."""
        return [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT_COMPLETION,
            ModelCapability.STREAMING,
        ]

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if not self.session:
            headers = {"Content-Type": "application/json"}
            # Add auth header if API key is provided (for hosted endpoints)
            if self.api_key and self.api_key != "local":
                headers["Authorization"] = f"Bearer {self.api_key}"

            self.session = aiohttp.ClientSession(headers=headers)
        return self.session

    def _convert_messages_to_prompt(self, messages: List[Message]) -> str:
        """Convert messages to Llama prompt format."""
        prompt_parts = []

        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")

        # Add final prompt for assistant response
        prompt_parts.append("Assistant:")

        return "\n\n".join(prompt_parts)

    def _convert_messages_for_ollama(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Convert messages to Ollama chat format."""
        ollama_messages = []

        for msg in messages:
            role = msg.role
            # Ollama uses 'system', 'user', 'assistant' roles
            if role not in ["system", "user", "assistant"]:
                role = "user"

            ollama_messages.append({"role": role, "content": msg.content})

        return ollama_messages

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def complete(self, messages: List[Message], model_config: ModelConfig) -> CompletionResponse:
        """Generate completion using Llama model."""
        if self._circuit_breaker_open:
            raise Exception(f"Circuit breaker open for {self.name}")

        session = await self._ensure_session()
        start_time = time.time()

        # Prepare request based on endpoint type
        if self.is_ollama:
            # Ollama API format
            endpoint = f"{self.base_url}/api/chat"
            payload = {
                "model": model_config.model_id,
                "messages": self._convert_messages_for_ollama(messages),
                "options": {
                    "temperature": model_config.temperature,
                    "top_p": model_config.top_p,
                    "num_predict": model_config.max_tokens,
                },
            }
            if model_config.stop_sequences:
                payload["options"]["stop"] = model_config.stop_sequences
        else:
            # Generic completion endpoint
            endpoint = f"{self.base_url}/completions"
            prompt = self._convert_messages_to_prompt(messages)
            payload = {
                "model": model_config.model_id,
                "prompt": prompt,
                "max_tokens": model_config.max_tokens,
                "temperature": model_config.temperature,
                "top_p": model_config.top_p,
            }
            if model_config.stop_sequences:
                payload["stop"] = model_config.stop_sequences

        try:
            async with session.post(
                endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=model_config.timeout)
            ) as response:
                response.raise_for_status()
                data = await response.json()

                latency_ms = (time.time() - start_time) * 1000

                # Extract response based on endpoint type
                if self.is_ollama:
                    content = data.get("message", {}).get("content", "")
                    # Ollama provides token counts in eval_count and prompt_eval_count
                    tokens_prompt = data.get("prompt_eval_count", 0)
                    tokens_completion = data.get("eval_count", 0)
                else:
                    # Generic response format
                    content = data.get("choices", [{}])[0].get("text", "")
                    # Estimate tokens if not provided
                    tokens_prompt = len(prompt.split()) * 1.3  # Rough estimate
                    tokens_completion = len(content.split()) * 1.3

                usage = {
                    "tokens_prompt": int(tokens_prompt),
                    "tokens_completion": int(tokens_completion),
                    "tokens_total": int(tokens_prompt + tokens_completion),
                }

                cost = self.estimate_cost(usage["tokens_prompt"], usage["tokens_completion"], model_config.model_id)

                self.update_metrics(success=True, tokens=usage["tokens_total"], cost=cost, latency_ms=latency_ms)

                return CompletionResponse(
                    content=content,
                    model=model_config.model_id,
                    usage=usage,
                    finish_reason="stop",
                    latency_ms=latency_ms,
                    provider=self.name,
                    metadata={
                        "total_duration": data.get("total_duration"),
                        "load_duration": data.get("load_duration"),
                        "eval_duration": data.get("eval_duration"),
                    },
                )

        except aiohttp.ClientResponseError as e:
            self._consecutive_failures += 1
            self.update_metrics(success=False, tokens=0, cost=0, latency_ms=0)

            if e.status == 404:
                raise Exception(
                    f"Model {model_config.model_id} not found. Please pull it first: ollama pull {model_config.model_id}"
                )
            else:
                raise Exception(f"API error from {self.name}: {e.status}")

        except Exception as e:
            self._consecutive_failures += 1
            self.update_metrics(success=False, tokens=0, cost=0, latency_ms=0)
            raise Exception(f"Error calling {self.name}: {str(e)}")

    async def stream_complete(self, messages: List[Message], model_config: ModelConfig) -> AsyncIterator[StreamChunk]:
        """Stream completion from Llama model."""
        if self._circuit_breaker_open:
            raise Exception(f"Circuit breaker open for {self.name}")

        session = await self._ensure_session()

        # Prepare request based on endpoint type
        if self.is_ollama:
            endpoint = f"{self.base_url}/api/chat"
            payload = {
                "model": model_config.model_id,
                "messages": self._convert_messages_for_ollama(messages),
                "stream": True,
                "options": {
                    "temperature": model_config.temperature,
                    "top_p": model_config.top_p,
                    "num_predict": model_config.max_tokens,
                },
            }
        else:
            endpoint = f"{self.base_url}/completions"
            prompt = self._convert_messages_to_prompt(messages)
            payload = {
                "model": model_config.model_id,
                "prompt": prompt,
                "max_tokens": model_config.max_tokens,
                "temperature": model_config.temperature,
                "top_p": model_config.top_p,
                "stream": True,
            }

        try:
            async with session.post(
                endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=model_config.timeout)
            ) as response:
                response.raise_for_status()

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        if self.is_ollama:
                            # Ollama streaming format
                            if "message" in data:
                                content = data["message"].get("content", "")
                                if content:
                                    yield StreamChunk(
                                        delta=content,
                                        finish_reason="stop" if data.get("done") else None,
                                        metadata={"model": data.get("model")},
                                    )

                            # Final message with stats
                            if data.get("done"):
                                yield StreamChunk(
                                    delta="",
                                    finish_reason="stop",
                                    usage={
                                        "tokens_prompt": data.get("prompt_eval_count", 0),
                                        "tokens_completion": data.get("eval_count", 0),
                                        "tokens_total": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                                    },
                                )
                        else:
                            # Generic streaming format
                            if "choices" in data:
                                choice = data["choices"][0]
                                if "text" in choice:
                                    yield StreamChunk(delta=choice["text"], finish_reason=choice.get("finish_reason"))

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            self._consecutive_failures += 1
            raise Exception(f"Streaming error from {self.name}: {str(e)}")

    async def generate_embeddings(self, texts: List[str], model: str = "llama3") -> List[List[float]]:
        """
        Generate embeddings using Llama.
        Note: This requires a model that supports embeddings.
        """
        if not self.is_ollama:
            raise NotImplementedError(
                "Embeddings are only supported with Ollama backend. "
                "Consider using sentence-transformers or another embedding model."
            )

        session = await self._ensure_session()
        embeddings = []

        for text in texts:
            payload = {
                "model": model,
                "prompt": text,
            }

            try:
                async with session.post(
                    f"{self.base_url}/api/embeddings", json=payload, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    embeddings.append(data["embedding"])

            except Exception as e:
                raise Exception(f"Embedding error from {self.name}: {str(e)}")

        return embeddings

    def estimate_cost(self, tokens_prompt: int, tokens_completion: int, model: str) -> float:
        """Estimate cost for Llama token usage (mainly compute costs for self-hosted)."""
        # For self-hosted models, cost is mainly compute resources
        # Use a default low cost
        base_model = model.split(":")[0] if ":" in model else model

        if base_model not in self.PRICING:
            # Default pricing for unknown models
            pricing = {"prompt": 0.0001, "completion": 0.0001}
        else:
            pricing = self.PRICING[base_model]

        prompt_cost = (tokens_prompt / 1000) * pricing["prompt"]
        completion_cost = (tokens_completion / 1000) * pricing["completion"]

        return prompt_cost + completion_cost

    async def list_models(self) -> List[str]:
        """List available models (Ollama specific)."""
        if not self.is_ollama:
            return self.supported_models

        session = await self._ensure_session()

        try:
            async with session.get(f"{self.base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=10)) as response:
                response.raise_for_status()
                data = await response.json()

                models = []
                for model in data.get("models", []):
                    models.append(model["name"])

                return models

        except Exception as e:
            return self.supported_models

    async def pull_model(self, model_name: str) -> bool:
        """Pull a model (Ollama specific)."""
        if not self.is_ollama:
            raise NotImplementedError("Model pulling is only supported with Ollama backend")

        session = await self._ensure_session()

        payload = {"name": model_name}

        try:
            async with session.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=3600),  # 1 hour timeout for large models
            ) as response:
                response.raise_for_status()

                # Stream the pull progress
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line:
                        data = json.loads(line)
                        if data.get("status") == "success":
                            return True

                return True

        except Exception as e:
            raise Exception(f"Error pulling model {model_name}: {str(e)}")

    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    def __del__(self):
        """Cleanup on deletion."""
        if self.session:
            asyncio.create_task(self.close())
