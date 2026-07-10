"""Provider interface and implementations.

Defines the Provider protocol that all providers must implement,
and standardizes inference using AsyncOpenAI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from openai import AsyncOpenAI
from config import ProviderConfig

logger = logging.getLogger(__name__)


@dataclass
class ProviderResponse:
    """Standard response format returned by all providers."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    provider: str


@runtime_checkable
class Provider(Protocol):
    """Protocol that all providers must satisfy.

    Providers perform inference only. They never make routing decisions.
    """

    async def generate(self, prompt: str, **kwargs: Any) -> ProviderResponse:
        """Generate a response for the given prompt asynchronously.

        Args:
            prompt: The user prompt to process.
            **kwargs: Additional provider-specific parameters.

        Returns:
            A ProviderResponse with the model's output.
        """
        ...


import os
import asyncio

class LocalProvider:
    """Local model provider.
    
    Supports:
    1. In-process inference using llama-cpp-python (GGUF) for standalone Docker deployment.
    2. HTTP inference using OpenAI-compatible API (e.g., LM Studio/Ollama) for development.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.name = "local"
        self.in_process_model = None

        # Check if the configured model is a local GGUF path
        is_gguf = config.model.endswith(".gguf") or os.path.exists(config.model)
        
        if is_gguf and os.path.exists(config.model):
            logger.info("Initializing in-process Llama provider from %s", config.model)
            try:
                from llama_cpp import Llama
                # Load the model with 4-bit quantization and CPU threads matching the 2 vCPU budget
                self.in_process_model = Llama(
                    model_path=config.model,
                    n_ctx=2048,
                    n_threads=2,
                    verbose=False
                )
                logger.info("In-process Llama provider initialized successfully.")
            except Exception as e:
                logger.error("Failed to load in-process Llama provider: %s. Falling back to HTTP client.", e)
        
        if self.in_process_model is None:
            self.client = AsyncOpenAI(
                base_url=config.base_url,
                api_key=config.api_key or "not-needed",
                timeout=config.timeout,
            )
            logger.info("LocalProvider (HTTP API) initialized: model=%s @ %s", config.model, config.base_url)

    async def generate(self, prompt: str, **kwargs: Any) -> ProviderResponse:
        """Send prompt to the local model."""
        if self.in_process_model is not None:
            # Run in-process Llama inference in a threadpool to prevent blocking the event loop
            loop = asyncio.get_running_loop()
            
            # Format using a standard prompt template for Gemma 2 IT
            formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            
            def _inference():
                return self.in_process_model(
                    formatted_prompt,
                    max_tokens=512,
                    temperature=0.3,
                    stop=["<end_of_turn>", "<start_of_turn>"],
                )
                
            response = await loop.run_in_executor(None, _inference)
            content = response["choices"][0]["text"]
            
            prompt_tokens = response["usage"]["prompt_tokens"]
            completion_tokens = response["usage"]["completion_tokens"]
            
            return ProviderResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=os.path.basename(self.config.model),
                provider=self.name,
            )
        else:
            messages = [{"role": "user", "content": prompt}]
            call_kwargs = {**kwargs, "model": self.config.model, "messages": messages}
            response = await self.client.chat.completions.create(**call_kwargs)
            
            content = response.choices[0].message.content or ""
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            
            return ProviderResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=self.config.model,
                provider=self.name,
            )


class RemoteProvider:
    """Remote model provider (e.g., Fireworks AI)."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.name = "remote"
        if not config.api_key:
            logger.warning("RemoteProvider initialized without an API key!")
            
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key or "missing-api-key",
            timeout=config.timeout,
        )
        logger.info("RemoteProvider initialized: model=%s", config.model)

    async def generate(self, prompt: str, **kwargs: Any) -> ProviderResponse:
        """Send prompt to the remote model."""
        messages = [{"role": "user", "content": prompt}]
        
        call_kwargs = {**kwargs, "model": self.config.model, "messages": messages}
        
        response = await self.client.chat.completions.create(**call_kwargs)
        
        content = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        
        return ProviderResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.config.model,
            provider=self.name,
        )
