"""
Pytest configuration for newspaper-explorer tests.

Defines custom markers and shared configuration.
Imports all fixtures from centralized mock_data module.
"""

from typing import TYPE_CHECKING

# Import all fixtures from centralized mock_data module
# This makes them available to all tests in the test suite
pytest_plugins = ["fixtures.data.mock_data"]

if TYPE_CHECKING:
    import pytest


def pytest_configure(config: "pytest.Config") -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (uses fixtures/real I/O)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (downloads models, runs inference, >10s)"
    )
