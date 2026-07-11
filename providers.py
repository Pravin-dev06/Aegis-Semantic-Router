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
import threading
from pathlib import Path

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
        # threading.Lock serializes llama-cpp calls — the Llama object is NOT thread-safe.
        # asyncio.gather() runs tasks concurrently via run_in_executor which spawns threads;
        # without this lock, concurrent inference calls crash with GGML_ASSERT(buffer) failed.
        self._llama_lock = threading.Lock()

        # Resolve model path: if config.model is overridden by ALLOWED_MODELS (e.g. accounts/fireworks/...)
        # but the local GGUF file is baked into the image at models/gemma-2-2b-it-Q4_K_M.gguf,
        # we must use the local GGUF file to guarantee local in-process execution.
        model_path = config.model
        default_gguf = "models/gemma-2-2b-it-Q4_K_M.gguf"
        
        if not os.path.exists(model_path):
            # Check if default GGUF or any GGUF exists in models/
            if os.path.exists(default_gguf):
                model_path = default_gguf
            else:
                # Scan models/ directory for any GGUF file
                models_dir = Path("models")
                if models_dir.exists():
                    gguf_files = list(models_dir.glob("*.gguf"))
                    if gguf_files:
                        model_path = str(gguf_files[0])

        is_gguf = model_path.endswith(".gguf") or os.path.exists(model_path)
        
        if is_gguf and os.path.exists(model_path):
            logger.info("Initializing in-process Llama provider from %s", model_path)
            try:
                from llama_cpp import Llama
                # Load the model with 4-bit quantization and CPU threads matching the 2 vCPU budget
                self.in_process_model = Llama(
                    model_path=model_path,
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
            
            # Format using the official Gemma 2 IT chat template.
            # System instruction goes BEFORE the first user turn using a <start_of_turn>user block.
            # This is the correct Gemma 2 format per official HuggingFace docs.
            system_prompt = (
                "You are a highly precise, direct, and concise assistant. "
                "Provide answers directly without conversational preamble or verbose explanation. "
                "For classification or extraction tasks, output ONLY the final answer (e.g. 'positive', 'negative', 'neutral')."
            )
            formatted_prompt = (
                f"<start_of_turn>user\n{system_prompt}\n\n{prompt}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            
            def _inference():
                # Acquire lock to prevent concurrent llama-cpp calls from crashing
                with self._llama_lock:
                    return self.in_process_model(
                        formatted_prompt,
                        max_tokens=512,
                        temperature=0.1,
                        stop=["<end_of_turn>", "<start_of_turn>"],
                    )
                
            response = await loop.run_in_executor(None, _inference)
            content = response["choices"][0]["text"].strip()
            
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
