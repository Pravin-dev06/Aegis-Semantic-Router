"""Tests for the benchmark module."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import BenchmarkReport, evaluate_response, run_benchmark
from config import ProviderConfig
from providers import LocalProvider, ProviderResponse, RemoteProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_dataset():
    return [
        {"id": 1, "prompt": "What is 2 + 2?", "expected": "4", "category": "math"},
        {"id": 2, "prompt": "Write def foo", "expected": "def foo", "category": "code"},
    ]


@pytest.fixture
def mock_local_provider():
    provider = MagicMock(spec=LocalProvider)
    provider.config = ProviderConfig(base_url="http://local/v1", model="test-local")
    provider.generate = AsyncMock()
    provider.generate.return_value = ProviderResponse(
        content="The answer is 4",
        prompt_tokens=0,
        completion_tokens=0,
        model="test-local",
        provider="local",
    )
    return provider


@pytest.fixture
def mock_remote_provider():
    provider = MagicMock(spec=RemoteProvider)
    provider.config = ProviderConfig(base_url="http://remote/v1", model="test-remote")
    provider.generate = AsyncMock()
    provider.generate.return_value = ProviderResponse(
        content="Here is def foo(): pass",
        prompt_tokens=10,
        completion_tokens=15,
        model="test-remote",
        provider="remote",
    )
    return provider


# ---------------------------------------------------------------------------
# evaluate_response
# ---------------------------------------------------------------------------

def test_evaluate_response_matching():
    assert evaluate_response("The capital is Paris.", "Paris") is True
    assert evaluate_response("paris is the capital", "Paris") is True
    assert evaluate_response("   London   ", "london") is True


def test_evaluate_response_mismatch():
    assert evaluate_response("The capital is London.", "Paris") is False
    assert evaluate_response("Rome is nice", "London") is False


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_benchmark_computes_correct_metrics(
    mock_dataset, mock_local_provider, mock_remote_provider, tmp_path
):
    dataset_json = json.dumps(mock_dataset)
    dataset_file = tmp_path / "test_dataset.json"
    dataset_file.write_text(dataset_json, encoding="utf-8")

    # threshold = 0.5
    # Prompt 1 ("What is 2 + 2?") -> complexity score < 0.5 -> local -> outputs "The answer is 4" -> expected is "4" -> PASS
    # Prompt 2 ("Write def foo") -> complexity score = 0.4 -> local -> outputs "The answer is 4" (because mock local returns that) -> expected is "def foo" -> FAIL
    # Let's run with threshold=0.3 to force Prompt 2 to go remote.
    # Prompt 2 ("Write def foo") complexity score: length (medium, 13 chars -> 0.0), code keywords ("def foo" -> 0.4). Total score = 0.4.
    # Since 0.4 > 0.3 threshold -> routes remote -> remote outputs "Here is def foo(): pass" -> expected is "def foo" -> PASS

    with patch("builtins.print") as mock_print:
        report = await run_benchmark(
            dataset_path=str(dataset_file),
            local_provider=mock_local_provider,
            remote_provider=mock_remote_provider,
            threshold=0.3,
            fallback_enabled=True,
            output_dir=str(tmp_path / "reports"),
        )

    # 2 cases, both PASS (since Prompt 1 goes local and gets 4, Prompt 2 goes remote and gets def foo)
    assert report.total_cases == 2
    assert report.correct_cases == 2
    assert report.accuracy == 1.0

    # Routing stats
    assert report.local_routing_count == 1
    assert report.remote_routing_count == 1
    assert report.local_routing_pct == 50.0
    assert report.remote_routing_pct == 50.0

    # Token calculations — only remote tokens are counted
    assert report.total_remote_prompt_tokens == 10
    assert report.total_remote_completion_tokens == 15
    assert report.total_remote_tokens == 25

    # Concurrency and file saving check
    reports_dir = tmp_path / "reports"
    saved_reports = list(reports_dir.glob("benchmark_*.json"))
    assert len(saved_reports) == 1
    with open(saved_reports[0], "r", encoding="utf-8") as f:
        saved_data = json.load(f)
        assert saved_data["total_cases"] == 2
        assert saved_data["accuracy"] == 1.0
        assert saved_data["total_remote_tokens"] == 25
