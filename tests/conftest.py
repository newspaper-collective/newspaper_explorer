"""
Pytest configuration for newspaper-explorer tests.

Defines custom markers and shared configuration.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (uses fixtures/real I/O)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (downloads models, runs inference, >10s)"
    )
