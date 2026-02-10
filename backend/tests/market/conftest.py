"""Fixtures for market data tests.

Ensures the ``massive`` package can be safely imported even when not installed,
so that MassiveDataSource tests can mock the REST client without requiring the
real dependency.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _mock_massive_if_missing():
    """Insert stub ``massive`` modules into sys.modules when the package is absent."""
    try:
        import massive  # noqa: F401

        yield  # Real package available — nothing to do
    except ImportError:
        # Create minimal stub modules so `from massive import RESTClient`
        # and `from massive.rest.models import SnapshotMarketType` resolve.
        massive_mod = types.ModuleType("massive")
        massive_mod.RESTClient = MagicMock  # type: ignore[attr-defined]

        rest_mod = types.ModuleType("massive.rest")
        models_mod = types.ModuleType("massive.rest.models")
        models_mod.SnapshotMarketType = MagicMock()  # type: ignore[attr-defined]

        massive_mod.rest = rest_mod  # type: ignore[attr-defined]
        rest_mod.models = models_mod  # type: ignore[attr-defined]

        stubs = {
            "massive": massive_mod,
            "massive.rest": rest_mod,
            "massive.rest.models": models_mod,
        }
        sys.modules.update(stubs)
        yield
        for key in stubs:
            sys.modules.pop(key, None)
