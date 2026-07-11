"""Tests for the provider implementations using AsyncOpenAI."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ProviderConfig
from providers import LocalProvider, RemoteProvider, ProviderResponse


@pytest.fixture
def mock_openai_client():
    """Fixture to mock AsyncOpenAI client."""
    with patch("providers.AsyncOpenAI") as mock:
        client_instance = AsyncMock()
        
        # Setup mock response structure
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "Mocked provider response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        
        client_instance.chat.completions.create.return_value = mock_response
        mock.return_value = client_instance
        
        yield client_instance


@pytest.mark.asyncio
async def test_local_provider_generate(mock_openai_client):
    """Test LocalProvider generate method uses HTTP path when no GGUF file is present."""
    cfg = ProviderConfig(base_url="http://localhost:1234/v1", model="local-model")

    # Patch os.path.exists and Path.glob so GGUF auto-discovery finds nothing,
    # forcing LocalProvider to use the HTTP/OpenAI client path (the mock).
    with patch("providers.os.path.exists", return_value=False), \
         patch("providers.Path.exists", return_value=False):
        provider = LocalProvider(cfg)

    response = await provider.generate("Hello local")

    assert isinstance(response, ProviderResponse)
    assert response.content == "Mocked provider response"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 20
    assert response.model == "local-model"
    assert response.provider == "local"

    mock_openai_client.chat.completions.create.assert_called_once_with(
        model="local-model",
        messages=[{"role": "user", "content": "Hello local"}]
    )


@pytest.mark.asyncio
async def test_remote_provider_generate(mock_openai_client):
    """Test RemoteProvider generate method."""
    cfg = ProviderConfig(
        base_url="https://api.fireworks.ai/inference/v1",
        model="remote-model",
        api_key="test-key"
    )
    provider = RemoteProvider(cfg)
    
    response = await provider.generate("Hello remote", temperature=0.7)
    
    assert isinstance(response, ProviderResponse)
    assert response.content == "Mocked provider response"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 20
    assert response.model == "remote-model"
    assert response.provider == "remote"
    
    mock_openai_client.chat.completions.create.assert_called_once_with(
        model="remote-model",
        messages=[{"role": "user", "content": "Hello remote"}],
        temperature=0.7
    )


def test_remote_provider_missing_key(caplog):
    """Test that a warning is logged when API key is missing for RemoteProvider."""
    cfg = ProviderConfig(base_url="https://api.fireworks.ai/inference/v1", model="test")
    provider = RemoteProvider(cfg)
    
    assert "without an API key" in caplog.text
