"""Transports: the two ways a request reaches the server.

Both are thin. The server takes a decoded JSON value and returns a decoded JSON
value, so a transport's only job is framing, validation of whatever the
transport itself adds, and handing the message over. Neither holds state between
requests, because the protocol has none to hold.
"""

from __future__ import annotations

from .http import MAX_BODY_BYTES, HttpResponse, StreamableHttp, serve_http
from .stdio import serve_stdio

__all__ = [
    "MAX_BODY_BYTES",
    "HttpResponse",
    "StreamableHttp",
    "serve_http",
    "serve_stdio",
]
