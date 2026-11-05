from __future__ import annotations

import dataclasses

import pytest
from fastapi.testclient import TestClient

from llm_firewall.config import Settings
from llm_firewall.pipeline import Firewall
from llm_firewall.proxy.app import create_app

API_KEY = "test-key-123"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return dataclasses.replace(
        Settings.from_env(),
        api_keys=frozenset({API_KEY}),
        upstream_base_url="mock",
        rate_limit_per_minute=600,
        rate_limit_burst=100,
        max_request_bytes=20_000,
        max_prompt_chars=5_000,
        log_level="WARNING",
        demo_mode=True,
    )


@pytest.fixture(scope="session")
def firewall(settings: Settings) -> Firewall:
    # Layer C is excluded so tests are deterministic whether or not a model is present.
    return Firewall(settings, load_transformer=False)


@pytest.fixture()
def client(settings: Settings, firewall: Firewall) -> TestClient:
    return TestClient(create_app(settings, firewall=firewall))


@pytest.fixture()
def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}
