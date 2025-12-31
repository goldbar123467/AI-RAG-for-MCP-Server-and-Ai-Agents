"""Pytest configuration for async tests."""

import pytest


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use default event loop policy."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()


# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)
