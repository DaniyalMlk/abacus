"""A reference client for revision ``2026-07-28``.

It exists to be driven by the conformance suite, and it is written to a rule
that makes it useful for that: **it implements the current revision and nothing
else.** There is no ``initialize`` call, no session header, no fallback to an
older shape, and no accommodation for a server that answers the way servers used
to. A client that quietly tolerated the old form would make a non-conforming
server look fine, which is the one thing a reference client must not do.

It is also deliberately thin on judgement. It builds well-formed requests, sends
them, and returns what came back, checking only the JSON-RPC framing that must
hold for a response to be matched to a request at all. Everything else —
``resultType``, cache hints, error-code allocation, which status code a
rejection deserves — is checked by :mod:`abacus.conformance`, because a client
that raised on those could not report on them.

Three transports, one interface:

:class:`InProcessTransport`
    Calls a server object directly. No framing, no sockets. Useful for driving
    the checks in a test suite.
:class:`StdioTransport`
    Launches a server as a subprocess and speaks line-delimited JSON at it,
    which is how an MCP client actually starts a local server.
:class:`HttpTransport`
    Posts to a Streamable HTTP endpoint, with the routing headers this revision
    requires and the Base64 sentinel where a value cannot travel as plain ASCII.
"""

from __future__ import annotations

import base64
import contextlib
import json
import subprocess
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Protocol

from .protocol import (
    META_CLIENT_CAPABILITIES,
    META_CLIENT_INFO,
    META_LOG_LEVEL,
    META_PROTOCOL_VERSION,
    PROTOCOL_VERSION,
)
from .transports.http import NAME_SOURCE

#: How long to wait for a server to answer, in seconds. A conformance run should
#: fail rather than hang when a server stops responding.
DEFAULT_TIMEOUT_S = 30.0


class TransportError(Exception):
    """The server could not be reached, or answered with something unreadable.

    Distinct from a protocol error, which is a well-formed answer saying no.
    """


@dataclass(frozen=True)
class Exchange:
    """One request and whatever came back, kept together.

    The raw response is retained alongside the parsed halves because the
    conformance checks ask questions about the envelope — whether ``id`` was
    echoed, whether an unexpected member is present — that a parsed view would
    have thrown away.
    """

    request: dict[str, Any]
    raw: Any
    status: int | None = None
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def result(self) -> dict[str, Any] | None:
        if isinstance(self.raw, dict) and isinstance(self.raw.get("result"), dict):
            body: dict[str, Any] = self.raw["result"]
            return body
        return None

    @property
    def error(self) -> dict[str, Any] | None:
        if isinstance(self.raw, dict) and isinstance(self.raw.get("error"), dict):
            body: dict[str, Any] = self.raw["error"]
            return body
        return None

    @property
    def ok(self) -> bool:
        return self.result is not None

    def code(self) -> int | None:
        error = self.error
        value = error.get("code") if error else None
        return value if isinstance(value, int) else None


def encode_header_value(value: str) -> str:
    """Render a value for a routing header, wrapping it if it cannot travel plainly.

    The inverse of :func:`abacus.transports.http.decode_header_value`. A value
    that is not printable ASCII, or that has significant leading or trailing
    whitespace, goes inside the ``=?base64?...?=`` sentinel; anything else is
    sent as it is, because wrapping unnecessarily makes traffic harder to read
    for no gain.
    """
    plain = all(0x20 <= ord(ch) < 0x7F for ch in value)
    if plain and value == value.strip():
        return value
    return "=?base64?" + base64.b64encode(value.encode()).decode("ascii") + "?="


class Transport(Protocol):
    """What the client needs from whatever carries its messages."""

    def send(self, payload: dict[str, Any], *, expect_response: bool) -> Exchange:
        """Send one message and return the exchange, empty for a notification."""

    def close(self) -> None:
        """Release whatever the transport holds."""


class InProcessTransport:
    """Hands messages to a server object directly.

    Everything a transport adds — framing, headers, status codes — is absent
    here by design, so a check that fails against this and passes against the
    others is a transport bug rather than a server one.
    """

    def __init__(self, server: Any) -> None:
        self.server = server

    def send(self, payload: dict[str, Any], *, expect_response: bool) -> Exchange:
        # Round-tripped through JSON so the client sees what a wire transport
        # would: tuples flattened to arrays, and no shared mutable objects
        # letting a server's response alias the request that produced it.
        decoded = json.loads(json.dumps(payload))
        raw = self.server.handle(decoded)
        if not expect_response:
            return Exchange(request=payload, raw=None)
        return Exchange(request=payload, raw=json.loads(json.dumps(raw)) if raw else None)

    def close(self) -> None:
        return None


class StdioTransport:
    """Launches a server and speaks line-delimited JSON at it over a pipe."""

    def __init__(
        self, command: Sequence[str], *, timeout_s: float = DEFAULT_TIMEOUT_S
    ) -> None:
        self.command = list(command)
        self.timeout_s = timeout_s
        self._process: subprocess.Popen[str] | None = None

    def _ensure(self) -> subprocess.Popen[str]:
        if self._process is None or self._process.poll() is not None:
            self._process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        return self._process

    def send(self, payload: dict[str, Any], *, expect_response: bool) -> Exchange:
        process = self._ensure()
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except BrokenPipeError as exc:
            raise TransportError(f"server exited: {self._stderr()}") from exc

        if not expect_response:
            return Exchange(request=payload, raw=None)

        line = process.stdout.readline()
        if not line:
            raise TransportError(
                f"server closed its output without answering: {self._stderr()}"
            )
        try:
            return Exchange(request=payload, raw=json.loads(line))
        except json.JSONDecodeError as exc:
            raise TransportError(f"server wrote a line that is not JSON: {line!r}") from exc

    def _stderr(self) -> str:
        process = self._process
        if process is None or process.stderr is None:
            return "(no output)"
        try:
            return process.stderr.read()[-2000:] or "(no output)"
        except OSError:  # pragma: no cover - only on an already-dead pipe
            return "(unreadable)"

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover
            process.kill()
        self._process = None

    def __enter__(self) -> StdioTransport:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.close()


class HttpTransport:
    """Posts to a Streamable HTTP endpoint with the headers this revision requires.

    Proxies are bypassed deliberately. An MCP endpoint is usually on loopback,
    and an ambient ``HTTP_PROXY`` reaching in front of it produces failures that
    look like conformance problems and are not.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.url = url
        self.timeout_s = timeout_s
        self.extra_headers = dict(extra_headers or {})
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def headers_for(self, payload: dict[str, Any]) -> dict[str, str]:
        """Build the routing headers that mirror this body.

        They mirror rather than duplicate: the server refuses a request whose
        headers disagree with its body, so these are derived from the payload
        and never passed in alongside it.
        """
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        params = payload.get("params")
        meta = params.get("_meta") if isinstance(params, dict) else None
        version = meta.get(META_PROTOCOL_VERSION) if isinstance(meta, dict) else None
        headers["MCP-Protocol-Version"] = (
            version if isinstance(version, str) else PROTOCOL_VERSION
        )
        method = payload.get("method")
        if isinstance(method, str):
            headers["Mcp-Method"] = method
            source = NAME_SOURCE.get(method)
            if source is not None and isinstance(params, dict):
                name = params.get(source)
                if isinstance(name, str):
                    headers["Mcp-Name"] = encode_header_value(name)
        headers.update(self.extra_headers)
        return headers

    def send(self, payload: dict[str, Any], *, expect_response: bool) -> Exchange:
        body = json.dumps(payload, separators=(",", ":")).encode()
        request = urllib.request.Request(
            self.url, data=body, headers=self.headers_for(payload), method="POST"
        )
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                return self._exchange(payload, response.status, response.headers, response.read())
        except urllib.error.HTTPError as exc:
            # A 4xx carrying a JSON-RPC error body is a normal answer in this
            # protocol, not a transport failure, so it is returned rather than
            # raised and the status is kept for the checks that care.
            return self._exchange(payload, exc.code, exc.headers, exc.read())
        except urllib.error.URLError as exc:
            raise TransportError(f"could not reach {self.url}: {exc.reason}") from exc

    @staticmethod
    def _exchange(
        payload: dict[str, Any], status: int, headers: Any, body: bytes
    ) -> Exchange:
        raw: Any = None
        if body:
            try:
                raw = json.loads(body)
            except json.JSONDecodeError:
                raw = {"_unparsed": body.decode(errors="replace")}
        return Exchange(
            request=payload,
            raw=raw,
            status=status,
            headers={k.lower(): v for k, v in dict(headers).items()},
        )

    def raw_request(
        self, method: str, *, headers: Mapping[str, str] | None = None, body: bytes = b""
    ) -> Exchange:
        """Send something deliberately malformed, for the checks that need to.

        The verbs this revision retired, and the header disagreements it
        requires a server to refuse, cannot be exercised through
        :meth:`send` — which is careful to be correct.
        """
        request = urllib.request.Request(
            self.url, data=body or None, headers=dict(headers or {}), method=method
        )
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                return self._exchange({}, response.status, response.headers, response.read())
        except urllib.error.HTTPError as exc:
            return self._exchange({}, exc.code, exc.headers, exc.read())
        except urllib.error.URLError as exc:
            raise TransportError(f"could not reach {self.url}: {exc.reason}") from exc

    def close(self) -> None:
        return None


class Client:
    """Drives a server over a transport, one request at a time.

    Request ids are monotonic integers. The protocol only requires that an id be
    unique among the requests in flight, and a counter makes an unmatched
    response obvious in a transcript, which a random id would not.
    """

    def __init__(
        self,
        transport: Transport,
        *,
        protocol_version: str = PROTOCOL_VERSION,
        client_name: str = "abacus-reference-client",
        client_version: str = "1.0.0",
        capabilities: Mapping[str, Any] | None = None,
    ) -> None:
        self.transport = transport
        self.protocol_version = protocol_version
        self.client_info = {"name": client_name, "version": client_version}
        self.capabilities: dict[str, Any] = dict(capabilities or {})
        self._next_id = 1

    # -- message construction ---------------------------------------------

    def meta(
        self,
        *,
        version: str | None = None,
        capabilities: Mapping[str, Any] | None = None,
        log_level: str | None = None,
        include_client_info: bool = True,
    ) -> dict[str, Any]:
        """Build the ``_meta`` block every request in this revision must carry."""
        block: dict[str, Any] = {
            META_PROTOCOL_VERSION: version if version is not None else self.protocol_version,
            META_CLIENT_CAPABILITIES: dict(
                self.capabilities if capabilities is None else capabilities
            ),
        }
        if include_client_info:
            block[META_CLIENT_INFO] = dict(self.client_info)
        if log_level is not None:
            block[META_LOG_LEVEL] = log_level
        return block

    def build(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        ident: str | int | None = None,
        meta: Mapping[str, Any] | None = None,
        notification: bool = False,
    ) -> dict[str, Any]:
        """Assemble a request or notification without sending it."""
        body: dict[str, Any] = dict(params or {})
        body["_meta"] = dict(meta) if meta is not None else self.meta()
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": body}
        if notification:
            return message
        if ident is None:
            ident = self._next_id
            self._next_id += 1
        message["id"] = ident
        return message

    # -- sending -----------------------------------------------------------

    def send(self, message: dict[str, Any]) -> Exchange:
        """Send a message that is already built, however malformed it may be."""
        exchange = self.transport.send(message, expect_response="id" in message)
        if "id" in message:
            self._check_framing(exchange)
        return exchange

    def request(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        ident: str | int | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> Exchange:
        return self.send(self.build(method, params, ident=ident, meta=meta))

    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> Exchange:
        return self.send(self.build(method, params, notification=True))

    def _check_framing(self, exchange: Exchange) -> None:
        """Check only what must hold for a response to be matched to its request.

        Anything beyond this is a conformance question rather than a transport
        one, and raising here would stop the suite from reporting on it.
        """
        raw = exchange.raw
        if not isinstance(raw, dict):
            raise TransportError(f"response is not a JSON object: {raw!r}")
        if raw.get("jsonrpc") != "2.0":
            raise TransportError(f'response is missing "jsonrpc": "2.0": {raw!r}')
        if ("result" in raw) == ("error" in raw):
            raise TransportError(
                f"a response carries exactly one of result and error: {raw!r}"
            )
        sent = exchange.request.get("id")
        if "id" in raw and raw["id"] != sent:
            raise TransportError(f"response id {raw['id']!r} does not match request {sent!r}")

    # -- the methods a client actually calls -------------------------------

    def discover(self, **kwargs: Any) -> Exchange:
        return self.request("server/discover", **kwargs)

    def list_tools(self, cursor: str | None = None, **kwargs: Any) -> Exchange:
        params = {} if cursor is None else {"cursor": cursor}
        return self.request("tools/list", params, **kwargs)

    def call_tool(
        self, name: str, arguments: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> Exchange:
        return self.request(
            "tools/call", {"name": name, "arguments": dict(arguments or {})}, **kwargs
        )

    def tool_names(self) -> list[str]:
        """Every tool the server offers, following pagination to the end."""
        names: list[str] = []
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            exchange = self.list_tools(cursor)
            result = exchange.result
            if result is None:
                raise TransportError(f"tools/list failed: {exchange.error}")
            for tool in result.get("tools", []):
                if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                    names.append(tool["name"])
            following = result.get("nextCursor")
            if not isinstance(following, str):
                return names
            if following in seen:
                # A server that returns a cursor it has already issued would
                # otherwise loop here forever.
                raise TransportError(f"tools/list repeated a cursor: {following!r}")
            seen.add(following)
            cursor = following

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.close()


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "Client",
    "Exchange",
    "HttpTransport",
    "InProcessTransport",
    "StdioTransport",
    "Transport",
    "TransportError",
    "encode_header_value",
]
