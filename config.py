"""Configuration loader using Pydantic for validation.

Loads config.yaml for application settings and .env for secrets.
All configurable values flow through this module.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProviderConfig(BaseModel):
    """Configuration for a single provider (local or remote)."""

    base_url: str
    model: str
    timeout: int = 30
    api_key: str = ""


class RoutingConfig(BaseModel):
    """Configuration for the routing engine."""

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_remote_tokens: int = Field(default=4096, gt=0)
    fallback_enabled: bool = True
    strategy: str = "heuristic"
    category_thresholds: dict[str, float] = Field(default_factory=dict)


class AppConfig(BaseModel):
    """Top-level application configuration."""

    local_provider: ProviderConfig
    remote_provider: ProviderConfig
    routing: RoutingConfig = RoutingConfig()


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load and validate configuration from YAML file and environment variables.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated AppConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the configuration is invalid.
    """
    load_dotenv()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    config = AppConfig(**data)

    # Apply Hackathon Evaluation Harness Overrides
    fw_api_key = os.getenv("FIREWORKS_API_KEY") or os.getenv("REMOTE_API_KEY")
    fw_base_url = os.getenv("FIREWORKS_BASE_URL")
    allowed_models_str = os.getenv("ALLOWED_MODELS")

    # 1. Override endpoints and keys
    if fw_api_key:
        config.local_provider.api_key = fw_api_key
        config.remote_provider.api_key = fw_api_key
    if fw_base_url:
        config.local_provider.base_url = fw_base_url
        config.remote_provider.base_url = fw_base_url

    # 2. Override models dynamically from ALLOWED_MODELS
    if allowed_models_str:
        allowed_models = [m.strip() for m in allowed_models_str.split(",") if m.strip()]
        if allowed_models:
            # First model in the list is the cheap/local option, last is the complex/remote option
            config.local_provider.model = allowed_models[0]
            config.remote_provider.model = allowed_models[-1]
            logger.info(
                "ALLOWED_MODELS override applied: local=%s, remote=%s",
                config.local_provider.model,
                config.remote_provider.model
            )

    logger.info(
        "Configuration loaded: local=%s, remote=%s",
        config.local_provider.model,
        config.remote_provider.model,
    )

    return config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    logger.info("Config validated successfully")
    logger.info("Local provider: %s @ %s", cfg.local_provider.model, cfg.local_provider.base_url)
    logger.info("Remote provider: %s @ %s", cfg.remote_provider.model, cfg.remote_provider.base_url)
    logger.info("Routing threshold: %s", cfg.routing.threshold)
