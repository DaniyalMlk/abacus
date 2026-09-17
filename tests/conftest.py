"""Shared helpers for building well-formed requests in tests."""

from __future__ import annotations

from typing import Any

import pytest

from abacus.protocol import (
    META_CLIENT_CAPABILITIES,
    META_CLIENT_INFO,
    META_PROTOCOL_VERSION,
    PROTOCOL_VERSION,
)
from abacus.server import Server, capabilities


def meta(
    *,
    version: str = PROTOCOL_VERSION,
    client_capabilities: dict[str, Any] | None = None,
    client_info: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a `_meta` block carrying everything a request is required to have.

    `extra` is a plain mapping rather than keyword arguments because the keys it
    carries are reserved `_meta` names containing dots and slashes, which are
    not valid Python identifiers.
    """
    block: dict[str, Any] = {
        META_PROTOCOL_VERSION: version,
        META_CLIENT_CAPABILITIES: {} if client_capabilities is None else client_capabilities,
    }
    if client_info is not None:
        block[META_CLIENT_INFO] = client_info
    if extra is not None:
        block.update(extra)
    return block


def request(
    method: str,
    params: dict[str, Any] | None = None,
    *,
    ident: str | int = 1,
    version: str = PROTOCOL_VERSION,
    client_capabilities: dict[str, Any] | None = None,
    client_info: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete JSON-RPC request with valid metadata."""
    body = dict(params or {})
    body["_meta"] = meta(
        version=version,
        client_capabilities=client_capabilities,
        client_info=client_info,
        extra=extra,
    )
    return {"jsonrpc": "2.0", "id": ident, "method": method, "params": body}


@pytest.fixture
def server() -> Server:
    """A bare server with one echo method, for exercising dispatch itself."""
    srv = Server(
        name="test-server",
        version="0.0.1",
        instructions="A server used by the test suite.",
        capabilities=capabilities(tools={"listChanged": False}),
    )

    @srv.method("test/echo")
    def _echo(req: Any) -> dict[str, Any]:
        from abacus.protocol import complete

        return complete(echoed=req.params.get("value"))

    @srv.method("test/needs-elicitation", requires=("elicitation",))
    def _needs(req: Any) -> dict[str, Any]:
        from abacus.protocol import complete

        return complete(ok=True)

    return srv
