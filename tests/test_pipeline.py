"""Integration smoke tests."""

import pytest


@pytest.mark.integration
def test_config_loads():
    from src.config.loader import get_settings
    s = get_settings()
    assert s.get("project", "name") == "RetailPulse"
