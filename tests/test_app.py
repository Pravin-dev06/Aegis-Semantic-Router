"""Tests for the FastAPI web server endpoints in app.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app

from providers import ProviderResponse
from router import RoutingResult

client = TestClient(app)


@pytest.fixture
def mock_routing_result():
    response = ProviderResponse(
        content="This is a mocked API prompt reply.",
        prompt_tokens=5,
        completion_tokens=10,
        model="local-model",
        provider="local",
    )
    return RoutingResult(
        response=response,
        provider_used="local",
        model_used="local-model",
        routing_reason="simple prompt",
        complexity_score=0.1,
        fallback_used=False,
    )


def test_serve_dashboard():
    """Verify that root GET path returns the main dashboard HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AMD ADAPTIVE" in response.text


@patch("app.global_config")
def test_get_config(mock_config):
    """Verify the config API exposes model settings."""
    mock_config.local_provider.model = "local-m"
    mock_config.remote_provider.model = "remote-m"
    mock_config.routing.threshold = 0.55

    response = client.get("/api/config")
    assert response.status_code == 200
    assert response.json() == {
        "local_model": "local-m",
        "remote_model": "remote-m",
        "threshold": 0.55,
    }


@patch("app.route", new_callable=AsyncMock)
@patch("app.local_provider_instance")
@patch("app.remote_provider_instance")
@patch("app.global_config")
def test_api_route_success(
    mock_config, mock_remote, mock_local, mock_route, mock_routing_result
):
    """Verify that POST route parses inputs and returns RouteResponse."""
    mock_config.routing.threshold = 0.5
    mock_config.routing.fallback_enabled = True
    mock_route.return_value = mock_routing_result

    payload = {"prompt": "Hello world", "threshold": 0.4}
    response = client.post("/api/route", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["provider_used"] == "local"
    assert data["model_used"] == "local-model"
    assert data["complexity_score"] == 0.1
    assert data["fallback_used"] is False
    assert data["response_content"] == "This is a mocked API prompt reply."

    # Verify our router function was invoked with the overridden threshold
    mock_route.assert_called_once_with(
        prompt="Hello world",
        local_provider=mock_local,
        remote_provider=mock_remote,
        threshold=0.4,
        fallback_enabled=True,
        category_thresholds=mock_config.routing.category_thresholds,
        strategy=mock_config.routing.strategy,
    )
