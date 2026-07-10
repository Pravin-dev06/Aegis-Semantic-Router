"""Tests for the router module — Phase 2/5 (Category-aware)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ProviderConfig
from providers import LocalProvider, Provider, ProviderResponse, RemoteProvider
from router import RoutingResult, analyze_prompt, route


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(provider: str = "local", model: str = "test-model") -> ProviderResponse:
    return ProviderResponse(
        content="Test response",
        prompt_tokens=5,
        completion_tokens=10,
        model=model,
        provider=provider,
    )


def _make_mock_provider(provider_name: str, raises: Exception | None = None) -> MagicMock:
    """Return a mock provider that satisfies the Provider protocol and has an async generate method."""
    mock = MagicMock(spec=Provider)
    mock_generate = AsyncMock()
    if raises:
        mock_generate.side_effect = raises
    else:
        mock_generate.return_value = _make_response(provider=provider_name)
    mock.generate = mock_generate
    return mock


# ---------------------------------------------------------------------------
# analyze_prompt
# ---------------------------------------------------------------------------

class TestAnalyzePrompt:
    """Test the feature extraction and category detection function in isolation."""

    def test_simple_short_prompt_scores_low(self):
        score, reason, category = analyze_prompt("What is the capital of France?")
        assert score <= 0.2
        assert "general" == category

    def test_long_prompt_increases_score(self):
        long_prompt = "a " * 300  # 600 chars
        score, reason, category = analyze_prompt(long_prompt)
        assert score >= 0.3
        assert "long prompt" in reason
        assert "general" == category

    def test_code_block_increases_score(self):
        score, reason, category = analyze_prompt("Write a function:\n```python\ndef foo(): pass\n```")
        assert score >= 0.4
        assert "code detected" in reason
        assert "programming" == category

    def test_math_keywords_increase_score(self):
        score, reason, category = analyze_prompt("Calculate the integral of x^2 from 0 to 10")
        assert score >= 0.3
        assert "math detected" in reason
        assert "mathematics" == category

    def test_reasoning_keywords_increase_score(self):
        score, reason, category = analyze_prompt("Explain the difference between TCP and UDP in detail.")
        assert score >= 0.2
        assert "reasoning/analysis keywords" in reason

    def test_combined_signals_cap_at_one(self):
        """Very complex prompt should never exceed 1.0."""
        complex_prompt = (
            "a " * 300 +
            "Calculate the integral ```def foo(): pass``` "
            "explain the difference between matrix and vector"
        )
        score, _, _ = analyze_prompt(complex_prompt)
        assert score == 1.0

    def test_score_is_within_bounds(self):
        for prompt in ["hi", "Hello world", "x " * 600, "def foo(): return 42"]:
            score, _, _ = analyze_prompt(prompt)
            assert 0.0 <= score <= 1.0

    def test_categories_detected_correctly(self):
        assert analyze_prompt("classify sentiment")[2] == "classification"
        assert analyze_prompt("summarize this passage")[2] == "summarization"
        assert analyze_prompt("solve 3x + 2 = 1")[2] == "mathematics"
        assert analyze_prompt("write a python algorithm")[2] == "programming"
        assert analyze_prompt("extract named entities")[2] == "ner"
        assert analyze_prompt("Three friends, Sam, Jo, and Lee, each own a different pet. Who owns the cat?")[2] == "logical_reasoning"


# ---------------------------------------------------------------------------
# Route — decision correctness
# ---------------------------------------------------------------------------

class TestRouteDecisions:
    """Test that route() picks the right provider based on complexity."""

    @pytest.mark.asyncio
    async def test_simple_prompt_goes_local(self):
        local = _make_mock_provider("local")
        remote = _make_mock_provider("remote")

        result = await route(
            "What is 2 + 2?",
            local_provider=local,
            remote_provider=remote,
            threshold=0.5,
        )

        assert result.provider_used == "local"
        assert result.fallback_used is False
        local.generate.assert_called_once()
        remote.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_complex_prompt_goes_remote(self):
        local = _make_mock_provider("local")
        remote = _make_mock_provider("remote")

        # Long + code + math → score will exceed 0.5
        complex_prompt = (
            "a " * 300 +
            "Write a Python class that calculates the integral of a function "
            "using Simpson's rule. ```python\nclass Integrator: pass\n```"
        )
        result = await route(
            complex_prompt,
            local_provider=local,
            remote_provider=remote,
            threshold=0.5,
        )

        assert result.provider_used == "remote"
        assert result.fallback_used is False
        remote.generate.assert_called_once()
        local.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_routing_result_has_metadata(self):
        local = _make_mock_provider("local")
        remote = _make_mock_provider("remote")

        result = await route(
            "What is the capital of France?",
            local_provider=local,
            remote_provider=remote,
        )

        assert isinstance(result, RoutingResult)
        assert isinstance(result.complexity_score, float)
        assert result.routing_reason != ""
        assert result.timestamp != ""
        assert result.category == "general"


# ---------------------------------------------------------------------------
# Category overrides & Strategy dispatch
# ---------------------------------------------------------------------------

class TestCategoryOverrides:
    """Test that route() respects per-category thresholds."""

    @pytest.mark.asyncio
    async def test_category_override_escalates(self):
        local = _make_mock_provider("local")
        remote = _make_mock_provider("remote")

        # A math prompt: "solve 2x=4"
        # Heuristic score for this prompt is 0.3 (math keyword)
        # With global threshold 0.5, it would normally go LOCAL
        # But with category_thresholds: math=0.2, 0.3 > 0.2, so it should go REMOTE.
        category_thresholds = {"mathematics": 0.2}

        result = await route(
            "solve 2x=4",
            local_provider=local,
            remote_provider=remote,
            threshold=0.5,
            category_thresholds=category_thresholds,
        )

        assert result.provider_used == "remote"
        assert result.threshold_used == 0.2
        remote.generate.assert_called_once()
        local.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_category_override_retains_local(self):
        local = _make_mock_provider("local")
        remote = _make_mock_provider("remote")

        # A summarization prompt: "summarize this long article about TCP UDP in detail"
        # Length + reasoning keywords + summarization keywords = high complexity score (e.g. 0.6)
        # With global threshold 0.5, it would normally go REMOTE.
        # But with category_thresholds: summarization=0.8, 0.6 <= 0.8, so it stays LOCAL.
        category_thresholds = {"summarization": 0.8}

        result = await route(
            "summarize this long article about TCP UDP in detail " + "a " * 100,
            local_provider=local,
            remote_provider=remote,
            threshold=0.5,
            category_thresholds=category_thresholds,
        )

        assert result.provider_used == "local"
        assert result.threshold_used == 0.8
        local.generate.assert_called_once()
        remote.generate.assert_not_called()


class TestRoutingStrategies:
    """Test strategy dispatcher and semantic routing logic (model is mocked)."""

    @pytest.mark.asyncio
    async def test_semantic_strategy_routes_local_when_score_low(self):
        """Semantic strategy sends low-complexity prompt to local provider."""
        local = _make_mock_provider("local")
        remote = _make_mock_provider("remote")

        # Patch _semantic_score so no model download is needed
        with patch("router._semantic_score", return_value=(0.1, "mocked: simple prompt")):
            result = await route(
                "What is 2 + 2?",
                local_provider=local,
                remote_provider=remote,
                threshold=0.5,
                strategy="semantic",
            )

        assert result.provider_used == "local"
        assert "mocked" in result.routing_reason
        local.generate.assert_called_once()
        remote.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_semantic_strategy_routes_remote_when_score_high(self):
        """Semantic strategy sends high-complexity prompt to remote provider."""
        local = _make_mock_provider("local")
        remote = _make_mock_provider("remote")

        with patch("router._semantic_score", return_value=(0.9, "mocked: complex code")):
            result = await route(
                "Write a binary search tree in Python.",
                local_provider=local,
                remote_provider=remote,
                threshold=0.5,
                strategy="semantic",
            )

        assert result.provider_used == "remote"
        remote.generate.assert_called_once()
        local.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_semantic_falls_back_to_heuristic_when_model_unavailable(self):
        """When the embedding model can't load, semantic silently falls back to heuristic."""
        local = _make_mock_provider("local")
        remote = _make_mock_provider("remote")

        with patch(
            "router._semantic_score",
            return_value=(0.5, "semantic model unavailable (fallback score 0.5)"),
        ):
            result = await route(
                "What is 2 + 2?",
                local_provider=local,
                remote_provider=remote,
                threshold=0.5,
                strategy="semantic",
            )

        # Falls back to heuristic — simple prompt stays local
        assert result.provider_used == "local"

    @pytest.mark.asyncio
    async def test_gatekeeper_strategy_falls_back_gracefully(self):
        """Gatekeeper strategy (still a stub) falls back to heuristic routing."""
        local = _make_mock_provider("local")
        remote = _make_mock_provider("remote")

        result = await route(
            "What is 2 + 2?",
            local_provider=local,
            remote_provider=remote,
            threshold=0.5,
            strategy="gatekeeper",
        )

        assert result.provider_used == "local"


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

class TestFallback:
    """Test the fallback mechanism when local provider fails."""

    @pytest.mark.asyncio
    async def test_local_failure_escalates_to_remote_when_fallback_enabled(self):
        local = _make_mock_provider("local", raises=ConnectionError("LM Studio not running"))
        remote = _make_mock_provider("remote")

        result = await route(
            "What is 2 + 2?",
            local_provider=local,
            remote_provider=remote,
            threshold=0.5,
            fallback_enabled=True,
        )

        assert result.provider_used == "remote"
        assert result.fallback_used is True
        assert "fallback" in result.routing_reason.lower()
        remote.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_local_failure_raises_when_fallback_disabled(self):
        local = _make_mock_provider("local", raises=ConnectionError("LM Studio not running"))
        remote = _make_mock_provider("remote")

        with pytest.raises(ConnectionError):
            await route(
                "What is 2 + 2?",
                local_provider=local,
                remote_provider=remote,
                threshold=0.5,
                fallback_enabled=False,
            )

        remote.generate.assert_not_called()


# ---------------------------------------------------------------------------
# Provider protocol (carried forward from Phase 0 / Phase 1)
# ---------------------------------------------------------------------------

class TestProviderProtocol:
    """Verify provider classes satisfy the Protocol."""

    def test_local_provider_is_provider(self):
        cfg = ProviderConfig(base_url="http://localhost/v1", model="test")
        provider = LocalProvider(cfg)
        assert isinstance(provider, Provider)

    def test_remote_provider_is_provider(self):
        cfg = ProviderConfig(base_url="http://remote/v1", model="test")
        provider = RemoteProvider(cfg)
        assert isinstance(provider, Provider)
