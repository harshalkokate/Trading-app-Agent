"""Tests for the SSE streaming router."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.market.cache import PriceCache
from app.market.stream import _generate_events, create_stream_router


def _fake_request(is_disconnected: bool = False) -> MagicMock:
    """Build a minimal Request stub for _generate_events."""
    request = MagicMock()
    request.client = MagicMock(host="127.0.0.1")
    request.is_disconnected = AsyncMock(return_value=is_disconnected)
    return request


async def _collect_events(
    cache: PriceCache,
    request: MagicMock,
    stop_after: int,
    interval: float = 0.01,
) -> list[str]:
    """Drive _generate_events until N chunks are received, then cancel."""
    gen = _generate_events(cache, request, interval=interval)
    chunks: list[str] = []
    try:
        async for chunk in gen:
            chunks.append(chunk)
            if len(chunks) >= stop_after:
                break
    finally:
        await gen.aclose()
    return chunks


class TestCreateStreamRouter:
    """Tests for the router factory."""

    def test_factory_returns_fresh_router(self):
        """Two factory calls must yield independent routers (no shared route registration)."""
        cache = PriceCache()
        r1 = create_stream_router(cache)
        r2 = create_stream_router(cache)
        assert r1 is not r2
        assert len(r1.routes) == 1
        assert len(r2.routes) == 1

    def test_router_exposes_prices_endpoint(self):
        """The router registers GET /api/stream/prices."""
        cache = PriceCache()
        router = create_stream_router(cache)
        paths = {route.path for route in router.routes}
        assert "/api/stream/prices" in paths


@pytest.mark.asyncio
class TestGenerateEvents:
    """Tests for the SSE async generator."""

    async def test_emits_retry_directive_first(self):
        """The first yielded chunk is the SSE retry directive."""
        cache = PriceCache()
        cache.update("AAPL", 190.50)
        chunks = await _collect_events(cache, _fake_request(), stop_after=2)
        assert chunks[0] == "retry: 1000\n\n"

    async def test_emits_cache_contents_as_sse_data(self):
        """After the retry directive, the stream emits a JSON data payload of the cache."""
        cache = PriceCache()
        cache.update("AAPL", 190.50)
        cache.update("GOOGL", 175.25)

        chunks = await _collect_events(cache, _fake_request(), stop_after=2)
        data_chunk = chunks[1]
        assert data_chunk.startswith("data: ")
        assert data_chunk.endswith("\n\n")

        payload = json.loads(data_chunk[len("data: "):-2])
        assert payload["AAPL"]["price"] == 190.50
        assert payload["GOOGL"]["price"] == 175.25
        assert payload["AAPL"]["ticker"] == "AAPL"

    async def test_emits_on_version_change(self):
        """A new cache update produces a new SSE data event."""
        cache = PriceCache()
        cache.update("AAPL", 190.00)

        gen = _generate_events(cache, _fake_request(), interval=0.01)
        chunks: list[str] = []
        try:
            # First two chunks: retry + initial data
            chunks.append(await asyncio.wait_for(gen.__anext__(), timeout=1.0))
            chunks.append(await asyncio.wait_for(gen.__anext__(), timeout=1.0))

            # No change yet — generator should skip (waiting on sleep/version)
            # Trigger an update, then expect a new data event
            cache.update("AAPL", 191.00)
            chunks.append(await asyncio.wait_for(gen.__anext__(), timeout=1.0))
        finally:
            await gen.aclose()

        assert chunks[0] == "retry: 1000\n\n"
        first_payload = json.loads(chunks[1][len("data: "):-2])
        second_payload = json.loads(chunks[2][len("data: "):-2])
        assert first_payload["AAPL"]["price"] == 190.00
        assert second_payload["AAPL"]["price"] == 191.00

    async def test_stops_when_client_disconnects(self):
        """The generator exits cleanly when the client disconnects."""
        cache = PriceCache()
        cache.update("AAPL", 190.00)

        request = _fake_request(is_disconnected=True)
        chunks: list[str] = []
        async for chunk in _generate_events(cache, request, interval=0.01):
            chunks.append(chunk)
            if len(chunks) > 5:
                pytest.fail("Generator did not stop on disconnect")

        # Only the retry directive is yielded before the disconnect check
        assert chunks == ["retry: 1000\n\n"]

    async def test_skips_empty_cache(self):
        """An empty cache yields only the retry directive; no data events."""
        cache = PriceCache()

        # Disconnect after a few ticks so the generator terminates without
        # further input. The generator must not yield any `data:` chunks
        # because the cache is empty.
        request = MagicMock()
        request.client = MagicMock(host="127.0.0.1")
        disconnect_calls = {"n": 0}

        async def is_disconnected() -> bool:
            disconnect_calls["n"] += 1
            return disconnect_calls["n"] > 3

        request.is_disconnected = is_disconnected

        chunks: list[str] = []
        async for chunk in _generate_events(cache, request, interval=0.01):
            chunks.append(chunk)

        assert chunks == ["retry: 1000\n\n"]
