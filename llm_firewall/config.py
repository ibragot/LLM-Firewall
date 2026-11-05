"""Runtime settings, read once from environment variables.

Why env vars? The Twelve-Factor App rule: config that differs between
deployments (keys, URLs, limits) lives in the environment, never in code.
That way a secret can never be committed by accident, and the same Docker
image / Vercel bundle runs anywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repository root (…/llm-firewall). Used to resolve relative paths so the app
# works no matter which directory it is started from (tests, Docker, Vercel).
ROOT_DIR = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p


@dataclass(frozen=True)
class Settings:
    # Upstream LLM
    upstream_base_url: str = "mock"
    upstream_api_key: str = ""
    upstream_model: str = "mock-llm-1"
    upstream_timeout_s: float = 30.0

    # Proxy protection
    api_keys: frozenset[str] = field(default_factory=frozenset)
    rate_limit_per_minute: int = 30
    rate_limit_burst: int = 10
    max_request_bytes: int = 32_768
    max_prompt_chars: int = 6_000
    request_timeout_s: float = 45.0

    # Detection / policy
    policy_path: Path = ROOT_DIR / "config" / "policy.yaml"
    linear_model_path: Path = ROOT_DIR / "models" / "linear_model.json"
    transformer_model_dir: Path = ROOT_DIR / "models" / "transformer"

    # Logging
    log_hash_salt: str = "dev-salt"
    log_level: str = "INFO"

    # Demo
    demo_mode: bool = True

    @property
    def use_mock_upstream(self) -> bool:
        return self.upstream_base_url.strip().lower() in {"", "mock"}

    @classmethod
    def from_env(cls) -> Settings:
        keys = {k.strip() for k in os.environ.get("FIREWALL_API_KEYS", "").split(",") if k.strip()}
        return cls(
            upstream_base_url=_env_str("UPSTREAM_BASE_URL", "mock"),
            upstream_api_key=_env_str("UPSTREAM_API_KEY", ""),
            upstream_model=_env_str("UPSTREAM_MODEL", "mock-llm-1"),
            upstream_timeout_s=_env_float("UPSTREAM_TIMEOUT_S", 30.0),
            api_keys=frozenset(keys),
            rate_limit_per_minute=_env_int("RATE_LIMIT_PER_MINUTE", 30),
            rate_limit_burst=_env_int("RATE_LIMIT_BURST", 10),
            max_request_bytes=_env_int("MAX_REQUEST_BYTES", 32_768),
            max_prompt_chars=_env_int("MAX_PROMPT_CHARS", 6_000),
            request_timeout_s=_env_float("REQUEST_TIMEOUT_S", 45.0),
            policy_path=_resolve(_env_str("POLICY_PATH", "config/policy.yaml")),
            linear_model_path=_resolve(_env_str("LINEAR_MODEL_PATH", "models/linear_model.json")),
            transformer_model_dir=_resolve(_env_str("TRANSFORMER_MODEL_DIR", "models/transformer")),
            log_hash_salt=_env_str("LOG_HASH_SALT", "dev-salt"),
            log_level=_env_str("LOG_LEVEL", "INFO"),
            demo_mode=_env_bool("DEMO_MODE", True),
        )
