"""The Streamable HTTP transport, as revision ``2026-07-28`` defines it.

This revision reshaped the transport, and most of what is written about
Streamable HTTP still describes the older form. Gone from this version:

- protocol-level sessions and the ``Mcp-Session-Id`` header,
- the standalone GET stream for server-initiated messages,
- SSE resumability through ``Last-Event-ID``,
- server-initiated JSON-RPC requests on a response stream.

Added: routing headers the server is required to validate against the body.

**Why validating those headers is a security matter rather than a formality.**
The transport mirrors selected body fields into headers so intermediaries can
route and rate-limit without parsing the body. That creates two sources of truth
for the same fact. If a load balancer routes on ``Mcp-Name: read_file`` while
the server executes ``params.name`` of ``delete_everything``, every control in
front of the server was applied to a request that is not the one that ran. So a
disagreement is refused outright with ``HeaderMismatch`` and ``400`` — and the
check must compare *decoded* values, or the sentinel encoding becomes the way
around it.

The handler is a pure function of method, path, headers and body, so every
rejection path can be tested without opening a socket.
"""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ..errors import (
    HEADER_MISMATCH,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    MISSING_REQUIRED_CLIENT_CAPABILITY,
    PARSE_ERROR,
    UNSUPPORTED_PROTOCOL_VERSION,
    HeaderMismatchError,
    ParseError,
)
from ..protocol import META_PROTOCOL_VERSION, error_response
from ..server import Server

#: Largest body the server will read. A transport that reads an unbounded body
#: is a memory-exhaustion vector that costs an attacker nothing.
MAX_BODY_BYTES = 4 * 1024 * 1024

#: Methods whose `Mcp-Name` header is required, and the params field it mirrors.
NAME_SOURCE: dict[str, str] = {
    "tools/call": "name",
    "resources/read": "uri",
    "prompts/get": "name",
}

#: Codes that must be answered with `400 Bad Request` rather than `200`.
_BAD_REQUEST_CODES = frozenset(
    {
        HEADER_MISMATCH,
        MISSING_REQUIRED_CLIENT_CAPABILITY,
        UNSUPPORTED_PROTOCOL_VERSION,
        INVALID_PARAMS,
        INVALID_REQUEST,
        PARSE_ERROR,
    }
)

_SENTINEL_PREFIX = "=?base64?"
_SENTINEL_SUFFIX = "?="


@dataclass
class HttpResponse:
    """A response, decoupled from whatever is going to write it to a socket."""

    status: int
    body: dict[str, Any] | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def encoded(self) -> bytes:
        if self.body is None:
            return b""
        return json.dumps(self.body, separators=(",", ":")).encode()


def decode_header_value(raw: str) -> str:
    """Decode a header value, unwrapping the Base64 sentinel if present.

    A value that cannot be carried as plain ASCII — non-ASCII characters,
    control characters, surrounding whitespace — travels as
    ``=?base64?<data>?=``. Decoding before comparison is required rather than
    optional: a server that compared the raw forms would see a mismatch between
    an encoded header and a plain body value and reject conforming traffic, and
    one that skipped the check entirely for encoded values would hand an
    attacker the way around it.
    """
    if raw.startswith(_SENTINEL_PREFIX) and raw.endswith(_SENTINEL_SUFFIX):
        payload = raw[len(_SENTINEL_PREFIX) : -len(_SENTINEL_SUFFIX)]
        try:
            return base64.b64decode(payload, validate=True).decode()
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise HeaderMismatchError(
                f"header value is marked Base64 but does not decode: {raw!r}"
            ) from exc
    return raw


def _values_agree(header: str, body: Any) -> bool:
    """Compare a header value with the body value it mirrors.

    Numbers are compared numerically rather than textually, so ``42`` and
    ``42.0`` agree — they are the same JSON number, and a client is free to
    render either.
    """
    if isinstance(body, bool):
        return header == ("true" if body else "false")
    if isinstance(body, int | float):
        try:
            return bool(float(header) == float(body))
        except ValueError:
            return False
    return bool(header == body)


class StreamableHttp:
    """Validates the HTTP layer and hands the message to the server."""

    def __init__(
        self,
        server: Server,
        *,
        allowed_origins: frozenset[str] | None = None,
        max_body_bytes: int = MAX_BODY_BYTES,
    ) -> None:
        self.server = server
        self.allowed_origins = allowed_origins
        self.max_body_bytes = max_body_bytes

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _lower(headers: dict[str, str]) -> dict[str, str]:
        # Field names are case-insensitive; field values are not.
        return {name.lower(): value for name, value in headers.items()}

    def _error(self, status: int, code: int, message: str, ident: Any = None) -> HttpResponse:
        return HttpResponse(status, error_response(ident, {"code": code, "message": message}))

    # -- validation -------------------------------------------------------

    def _check_origin(self, headers: dict[str, str]) -> HttpResponse | None:
        """Refuse a cross-origin request from a browser.

        Without this, a page the user visits can drive a local MCP server
        through DNS rebinding. An absent ``Origin`` is not a browser request and
        is allowed; a present and unrecognised one is refused.
        """
        origin = headers.get("origin")
        if origin is None or self.allowed_origins is None:
            return None
        if origin in self.allowed_origins:
            return None
        return HttpResponse(
            403,
            error_response(
                None,
                {"code": INVALID_REQUEST, "message": f"origin not allowed: {origin}"},
            ),
        )

    def _check_headers(self, headers: dict[str, str], payload: Any) -> HttpResponse | None:
        """Check the routing headers against the body they mirror."""
        if not isinstance(payload, dict):
            return None

        declared = headers.get("mcp-protocol-version")
        if declared is None:
            return self._error(
                400, HEADER_MISMATCH, "missing required header: MCP-Protocol-Version"
            )
        params = payload.get("params")
        in_body = None
        if isinstance(params, dict):
            meta = params.get("_meta")
            if isinstance(meta, dict):
                in_body = meta.get(META_PROTOCOL_VERSION)
        if isinstance(in_body, str) and declared != in_body:
            return self._error(
                400,
                HEADER_MISMATCH,
                f"header mismatch: MCP-Protocol-Version header value {declared!r} does not "
                f"match body value {in_body!r}",
                payload.get("id"),
            )

        method = payload.get("method")
        declared_method = headers.get("mcp-method")
        if declared_method is None:
            return self._error(400, HEADER_MISMATCH, "missing required header: Mcp-Method")
        if declared_method != method:
            return self._error(
                400,
                HEADER_MISMATCH,
                f"header mismatch: Mcp-Method header value {declared_method!r} does not "
                f"match body method {method!r}",
                payload.get("id"),
            )

        source = NAME_SOURCE.get(method) if isinstance(method, str) else None
        if source is not None:
            raw = headers.get("mcp-name")
            if raw is None:
                return self._error(
                    400, HEADER_MISMATCH, f"missing required header: Mcp-Name (for {method})"
                )
            try:
                declared_name = decode_header_value(raw)
            except HeaderMismatchError as exc:
                return self._error(400, HEADER_MISMATCH, exc.message, payload.get("id"))
            body_name = params.get(source) if isinstance(params, dict) else None
            if body_name is not None and not _values_agree(declared_name, body_name):
                return self._error(
                    400,
                    HEADER_MISMATCH,
                    f"header mismatch: Mcp-Name header value {declared_name!r} does not "
                    f"match body value {body_name!r}",
                    payload.get("id"),
                )

        return self._check_param_headers(headers, payload, params)

    def _check_param_headers(
        self, headers: dict[str, str], payload: Any, params: Any
    ) -> HttpResponse | None:
        """Check any ``Mcp-Param-*`` headers against the arguments they mirror."""
        arguments = params.get("arguments") if isinstance(params, dict) else None
        for name, raw in headers.items():
            if not name.startswith("mcp-param-"):
                continue
            key = name[len("mcp-param-") :]
            try:
                value = decode_header_value(raw)
            except HeaderMismatchError as exc:
                return self._error(400, HEADER_MISMATCH, exc.message, payload.get("id"))
            if not isinstance(arguments, dict):
                continue
            match = next((k for k in arguments if k.lower() == key.lower()), None)
            if match is None:
                continue
            if not _values_agree(value, arguments[match]):
                return self._error(
                    400,
                    HEADER_MISMATCH,
                    f"header mismatch: Mcp-Param-{key} header value {value!r} does not "
                    f"match argument {match!r}",
                    payload.get("id"),
                )
        return None

    # -- entry point ------------------------------------------------------

    def handle(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        *,
        endpoint: str = "/mcp",
    ) -> HttpResponse:
        """Handle one HTTP request. Pure: no sockets, no state."""
        lowered = self._lower(headers)

        refusal = self._check_origin(lowered)
        if refusal is not None:
            return refusal

        if path.split("?")[0] != endpoint:
            return self._error(404, METHOD_NOT_FOUND, f"no MCP endpoint at {path}")

        if method in ("GET", "DELETE"):
            # Both belonged to the older transport: GET opened the standalone
            # stream and DELETE terminated a session. Neither exists now, and
            # 405 is what tells an older client so.
            return HttpResponse(405, None, {"Allow": "POST"})

        if method != "POST":
            return HttpResponse(405, None, {"Allow": "POST"})

        if len(body) > self.max_body_bytes:
            return self._error(
                413, INVALID_REQUEST, f"body exceeds {self.max_body_bytes} bytes"
            )

        accept = lowered.get("accept", "")
        if accept and not ("application/json" in accept or "*/*" in accept):
            return self._error(
                406,
                INVALID_REQUEST,
                "Accept must include application/json and text/event-stream",
            )

        try:
            payload = json.loads(body.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            parse_error = ParseError(f"Parse error: {exc}")
            return HttpResponse(400, error_response(None, parse_error.to_dict()))

        refusal = self._check_headers(lowered, payload)
        if refusal is not None:
            return refusal

        response = self.server.handle(payload)
        if response is None:
            # A notification the server accepted. 202 with no body.
            return HttpResponse(202)

        error = response.get("error")
        if error is None:
            return HttpResponse(200, response)

        code = error.get("code")
        if code == METHOD_NOT_FOUND:
            # 404 with a JSON-RPC body, which is what lets a client tell a
            # modern server that lacks the method from a legacy endpoint that
            # was never there.
            return HttpResponse(404, response)
        if code in _BAD_REQUEST_CODES:
            return HttpResponse(400, response)
        return HttpResponse(200, response)


def serve_http(
    server: Server,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    endpoint: str = "/mcp",
    allowed_origins: frozenset[str] | None = None,
) -> ThreadingHTTPServer:
    """Build an HTTP server bound to ``host``.

    The default is loopback rather than ``0.0.0.0``. Binding every interface by
    default would expose an unauthenticated analytics server to the network the
    moment someone ran it on a laptop with the wrong Wi-Fi joined.
    """
    transport = StreamableHttp(server, allowed_origins=allowed_origins)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"abacus/{server.version}"

        def _dispatch(self, verb: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            result = transport.handle(
                verb, self.path, dict(self.headers.items()), raw, endpoint=endpoint
            )
            encoded = result.encoded()
            self.send_response(result.status)
            for name, value in result.headers.items():
                self.send_header(name, value)
            if encoded:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            if encoded:
                self.wfile.write(encoded)

        def do_POST(self) -> None:
            self._dispatch("POST")

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_DELETE(self) -> None:
            self._dispatch("DELETE")

        def log_message(self, format: str, *args: Any) -> None:
            # Silence the default stderr access log; the stdio transport shares
            # this process and its diagnostics belong there uncluttered.
            return

    return ThreadingHTTPServer((host, port), Handler)
