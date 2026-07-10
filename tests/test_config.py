"""Tests for the configuration system."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AppConfig, ProviderConfig, RoutingConfig, load_config


class TestProviderConfig:
    """Test ProviderConfig validation."""

    def test_valid_provider(self):
        cfg = ProviderConfig(base_url="http://localhost:1234/v1", model="test-model")
        assert cfg.base_url == "http://localhost:1234/v1"
        assert cfg.model == "test-model"
        assert cfg.timeout == 30  # default
        assert cfg.api_key == ""  # default

    def test_custom_timeout(self):
        cfg = ProviderConfig(base_url="http://localhost/v1", model="m", timeout=120)
        assert cfg.timeout == 120

    def test_missing_required_field(self):
        with pytest.raises(Exception):
            ProviderConfig(base_url="http://localhost/v1")  # missing model


class TestRoutingConfig:
    """Test RoutingConfig validation."""

    def test_defaults(self):
        cfg = RoutingConfig()
        assert cfg.threshold == 0.5
        assert cfg.max_remote_tokens == 4096
        assert cfg.fallback_enabled is True

    def test_threshold_bounds(self):
        cfg = RoutingConfig(threshold=0.0)
        assert cfg.threshold == 0.0

        cfg = RoutingConfig(threshold=1.0)
        assert cfg.threshold == 1.0

        with pytest.raises(Exception):
            RoutingConfig(threshold=-0.1)

        with pytest.raises(Exception):
            RoutingConfig(threshold=1.1)

    def test_max_remote_tokens_positive(self):
        with pytest.raises(Exception):
            RoutingConfig(max_remote_tokens=0)


class TestAppConfig:
    """Test full AppConfig construction."""

    def test_valid_config(self):
        cfg = AppConfig(
            local_provider=ProviderConfig(base_url="http://local/v1", model="local-m"),
            remote_provider=ProviderConfig(base_url="http://remote/v1", model="remote-m"),
        )
        assert cfg.local_provider.model == "local-m"
        assert cfg.remote_provider.model == "remote-m"
        assert cfg.routing.threshold == 0.5  # default routing


class TestLoadConfig:
    """Test config loading from YAML files."""

    def test_load_from_yaml(self, tmp_path):
        config_data = {
            "local_provider": {"base_url": "http://localhost:1234/v1", "model": "test-local"},
            "remote_provider": {"base_url": "http://remote:8080/v1", "model": "test-remote"},
            "routing": {"threshold": 0.7, "max_remote_tokens": 2048, "fallback_enabled": False},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        cfg = load_config(str(config_file))
        assert cfg.local_provider.model == "test-local"
        assert cfg.remote_provider.model == "test-remote"
        assert cfg.routing.threshold == 0.7
        assert cfg.routing.fallback_enabled is False

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_empty_file(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_config(str(config_file))

    def test_env_var_overrides_api_key(self, tmp_path):
        config_data = {
            "local_provider": {"base_url": "http://local/v1", "model": "m"},
            "remote_provider": {"base_url": "http://remote/v1", "model": "m"},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch.dict(os.environ, {"REMOTE_API_KEY": "secret-key-123"}):
            cfg = load_config(str(config_file))
            assert cfg.remote_provider.api_key == "secret-key-123"

    def test_default_routing_when_omitted(self, tmp_path):
        config_data = {
            "local_provider": {"base_url": "http://local/v1", "model": "m"},
            "remote_provider": {"base_url": "http://remote/v1", "model": "m"},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        cfg = load_config(str(config_file))
        assert cfg.routing.threshold == 0.5
        assert cfg.routing.fallback_enabled is True
