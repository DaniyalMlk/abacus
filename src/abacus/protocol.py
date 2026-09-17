"""JSON-RPC message types and the per-request metadata MCP layers on top.

Revision ``2026-07-28`` removed the ``initialize`` handshake. There is no
session, and a server may not infer anything from the connection a request
arrived on: every request carries its own protocol version, client identity and
capabilities in ``_meta``. Two requests on the same stdio pipe may come from
unrelated conversations, so this module is written so that nothing here can hold
state between requests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Final

from .errors import InvalidParams, InvalidRequest, UnsupportedProtocolVersionError

#: The revision this server implements.
PROTOCOL_VERSION: Final = "2026-07-28"

#: Every revision this server accepts, newest first. Earlier revisions used an
#: ``initialize`` handshake and a per-connection session; supporting one would
#: mean carrying a second, stateful code path, so the server is modern-only and
#: says so in the error it returns.
SUPPORTED_VERSIONS: Final = (PROTOCOL_VERSION,)

# Reserved `_meta` keys used by the core protocol.
META_PROTOCOL_VERSION: Final = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_INFO: Final = "io.modelcontextprotocol/clientInfo"
META_CLIENT_CAPABILITIES: Final = "io.modelcontextprotocol/clientCapabilities"
META_LOG_LEVEL: Final = "io.modelcontextprotocol/logLevel"
META_SERVER_INFO: Final = "io.modelcontextprotocol/serverInfo"
META_SUBSCRIPTION_ID: Final = "io.modelcontextprotocol/subscriptionId"
META_PROGRESS_TOKEN: Final = "progressToken"

#: Trace-context keys are exempt from the prefix rule, for compatibility with
#: the OpenTelemetry semantic conventions.
TRACE_CONTEXT_KEYS: Final = frozenset({"traceparent", "tracestate", "baggage"})

#: Log levels, from RFC 5424, ordered least to most severe.
LOG_LEVELS: Final = (
    "debug",
    "info",
    "notice",
    "warning",
    "error",
    "critical",
    "alert",
    "emergency",
)

_LABEL = re.compile(r"^[A-Za-z](?:[A-Za-z0-9-]*[A-Za-z0-9])?$")
_META_NAME = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


def split_meta_key(key: str) -> tuple[str | None, str]:
    """Split a ``_meta`` key into its optional prefix and its name.

    A prefix is a dot-separated series of labels followed by a slash. Only the
    first slash separates them, so a name may not itself contain one.
    """
    head, slash, tail = key.partition("/")
    if not slash:
        return None, key
    return head, tail


def is_valid_meta_key(key: str) -> bool:
    """Return whether ``key`` satisfies the naming rules for a ``_meta`` key.

    The name must begin and end with an alphanumeric character and may contain
    hyphens, underscores and dots in between; an empty name is allowed. Each
    prefix label must start with a letter, end with a letter or digit, and
    contain only letters, digits and hyphens in between.
    """
    if key in TRACE_CONTEXT_KEYS:
        return True
    prefix, name = split_meta_key(key)
    if name and not _META_NAME.match(name):
        return False
    if prefix is None:
        return True
    labels = prefix.split(".")
    return all(_LABEL.match(label) for label in labels)


def is_reserved_meta_key(key: str) -> bool:
    """Return whether ``key`` sits under a prefix the specification reserves.

    A prefix is reserved when its *second* label is ``modelcontextprotocol`` or
    ``mcp``. The second label rather than the first, because prefixes are
    written in reverse-DNS order: ``io.modelcontextprotocol/`` and ``dev.mcp/``
    are reserved, whereas ``com.example.mcp/`` — where the organisation is
    ``example`` and ``mcp`` is merely the last component — is not.

    Trace-context keys are reserved too, despite having no prefix at all.
    """
    if key in TRACE_CONTEXT_KEYS:
        return True
    prefix, _ = split_meta_key(key)
    if prefix is None:
        return False
    labels = prefix.split(".")
    return len(labels) >= 2 and labels[1] in ("modelcontextprotocol", "mcp")


@dataclass(frozen=True)
class Implementation:
    """The self-reported name and version of a client or server.

    Self-reported and unverified: useful for display, logging and debugging, and
    never for a security or behavioural decision.
    """

    name: str
    version: str
    title: str | None = None

    @classmethod
    def from_dict(cls, raw: object, *, where: str) -> Implementation:
        if not isinstance(raw, dict):
            raise InvalidParams(f"{where} must be an object")
        name = raw.get("name")
        version = raw.get("version")
        if not isinstance(name, str) or not name:
            raise InvalidParams(f"{where}.name must be a non-empty string")
        if not isinstance(version, str) or not version:
            raise InvalidParams(f"{where}.version must be a non-empty string")
        title = raw.get("title")
        if title is not None and not isinstance(title, str):
            raise InvalidParams(f"{where}.title must be a string when present")
        return cls(name=name, version=version, title=title)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "version": self.version}
        if self.title is not None:
            out["title"] = self.title
        return out


@dataclass(frozen=True)
class Capabilities:
    """A capability map, as declared by a client or advertised by a server.

    Capability names are kept as given rather than coerced into an enumeration.
    A client may declare a capability this server has never heard of, and
    discarding it at the boundary would make that invisible; the raw map is kept
    so that :meth:`declares` answers honestly about anything.
    """

    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: object, *, where: str) -> Capabilities:
        if not isinstance(raw, dict):
            raise InvalidParams(f"{where} must be an object")
        for key, value in raw.items():
            if not isinstance(key, str):
                raise InvalidParams(f"{where} keys must be strings")
            if value is not None and not isinstance(value, dict):
                raise InvalidParams(f"{where}.{key} must be an object when present")
        return cls(raw=dict(raw))

    def declares(self, name: str) -> bool:
        """Return whether ``name`` was declared.

        Supports dotted paths so a nested capability can be asked about
        directly, as in ``declares("elicitation.form")``.
        """
        node: Any = self.raw
        for part in name.split("."):
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        return True

    def extensions(self) -> dict[str, Any]:
        """Return the declared extension map, which may be empty.

        Tasks, MCP Apps and enterprise authorization all live outside the core
        protocol in this revision and are negotiated through here.
        """
        ext = self.raw.get("extensions")
        return dict(ext) if isinstance(ext, dict) else {}

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass(frozen=True)
class RequestMeta:
    """The protocol metadata carried by a single request.

    This replaces what the handshake used to establish once per session. It is
    frozen because nothing downstream has any business mutating the caller's
    declared version or capabilities.
    """

    protocol_version: str
    client_capabilities: Capabilities
    client_info: Implementation | None = None
    log_level: str | None = None
    progress_token: str | int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_logs(self) -> bool:
        """Whether the caller opted this request into log notifications.

        ``logging/setLevel`` is gone; the level is per-request, and a server must
        not emit ``notifications/message`` for a request that did not ask for
        them. Silence is the default.
        """
        return self.log_level is not None

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> RequestMeta:
        """Extract and validate ``_meta`` from a request's params.

        Raises :class:`InvalidParams` when a required field is absent or
        malformed, which the transport renders as ``-32602`` (and, on HTTP, a
        ``400``).
        """
        raw = params.get("_meta")
        if raw is None:
            raise InvalidParams(
                "request is missing _meta; every request must carry "
                f"{META_PROTOCOL_VERSION} and {META_CLIENT_CAPABILITIES}"
            )
        if not isinstance(raw, dict):
            raise InvalidParams("_meta must be an object")

        for key in raw:
            if not isinstance(key, str) or not is_valid_meta_key(key):
                raise InvalidParams(f"_meta key is not a valid key name: {key!r}")

        version = raw.get(META_PROTOCOL_VERSION)
        if not isinstance(version, str) or not version:
            raise InvalidParams(f"_meta.{META_PROTOCOL_VERSION} is required and must be a string")

        if META_CLIENT_CAPABILITIES not in raw:
            raise InvalidParams(f"_meta.{META_CLIENT_CAPABILITIES} is required")
        capabilities = Capabilities.from_dict(
            raw[META_CLIENT_CAPABILITIES], where=f"_meta.{META_CLIENT_CAPABILITIES}"
        )

        client_info = None
        if META_CLIENT_INFO in raw:
            client_info = Implementation.from_dict(
                raw[META_CLIENT_INFO], where=f"_meta.{META_CLIENT_INFO}"
            )

        log_level = raw.get(META_LOG_LEVEL)
        if log_level is not None and log_level not in LOG_LEVELS:
            raise InvalidParams(f"_meta.{META_LOG_LEVEL} must be one of: {', '.join(LOG_LEVELS)}")
        if log_level is not None and not isinstance(log_level, str):  # pragma: no cover
            raise InvalidParams(f"_meta.{META_LOG_LEVEL} must be a string")

        token = raw.get(META_PROGRESS_TOKEN)
        if token is not None and not isinstance(token, str | int):
            raise InvalidParams(f"_meta.{META_PROGRESS_TOKEN} must be a string or integer")
        if isinstance(token, bool):
            raise InvalidParams(f"_meta.{META_PROGRESS_TOKEN} must be a string or integer")

        return cls(
            protocol_version=version,
            client_capabilities=capabilities,
            client_info=client_info,
            log_level=log_level,
            progress_token=token,
            raw=dict(raw),
        )

    def require_version(self, supported: tuple[str, ...] = SUPPORTED_VERSIONS) -> None:
        """Raise unless this request's protocol version is one we implement."""
        if self.protocol_version not in supported:
            raise UnsupportedProtocolVersionError(self.protocol_version, list(supported))


@dataclass(frozen=True)
class Request:
    """A parsed JSON-RPC request, with its MCP metadata already validated."""

    id: str | int
    method: str
    params: dict[str, Any]
    meta: RequestMeta

    def argument(self, name: str) -> Any:
        return self.params.get(name)


@dataclass(frozen=True)
class Notification:
    """A parsed JSON-RPC notification. No response is ever sent for one."""

    method: str
    params: dict[str, Any]


def parse_message(payload: object) -> Request | Notification:
    """Parse a decoded JSON value into a request or a notification.

    Rejects batches: this revision has no batching, and silently processing the
    first element of a list would be worse than refusing it.
    """
    if isinstance(payload, list):
        raise InvalidRequest("batch requests are not part of this protocol revision")
    if not isinstance(payload, dict):
        raise InvalidRequest("a JSON-RPC message must be an object")
    if payload.get("jsonrpc") != "2.0":
        raise InvalidRequest('a JSON-RPC message must carry "jsonrpc": "2.0"')

    method = payload.get("method")
    if not isinstance(method, str) or not method:
        raise InvalidRequest("a JSON-RPC message must carry a non-empty string method")

    raw_params = payload.get("params", {})
    if raw_params is None:
        raw_params = {}
    if not isinstance(raw_params, dict):
        # Positional parameters are legal JSON-RPC but no MCP method uses them,
        # and accepting them would leave `_meta` with nowhere to live.
        raise InvalidRequest("params must be an object; positional params are not supported")

    if "id" not in payload:
        return Notification(method=method, params=raw_params)

    ident = payload["id"]
    if ident is None:
        raise InvalidRequest("a request id must not be null")
    if isinstance(ident, bool) or not isinstance(ident, str | int):
        raise InvalidRequest("a request id must be a string or an integer")

    return Request(
        id=ident,
        method=method,
        params=raw_params,
        meta=RequestMeta.from_params(raw_params),
    )


def complete(**fields: Any) -> dict[str, Any]:
    """Build an ordinary result.

    Every result carries ``resultType``. A client reading a result from an
    older server that omits the field must treat it as ``complete``, but a
    server implementing this revision always writes it.
    """
    return {"resultType": "complete", **fields}


def input_required(
    input_requests: dict[str, Any], *, request_state: str | None = None
) -> dict[str, Any]:
    """Build an interim result asking the caller for more information.

    This is the multi round-trip pattern that replaced server-initiated
    requests. Rather than the server sending its own request down the stream, it
    answers with the questions it needs answered; the client gathers the input
    and retries the *original* request — with a fresh JSON-RPC id — carrying
    ``inputResponses`` and echoing ``requestState`` back.

    Because there is no session to hold the half-finished call, whatever the
    server needs to resume must be encoded into ``request_state`` and handed to
    the client. Anything the server keeps on its own side instead has to be
    keyed by something the client will send back.
    """
    result: dict[str, Any] = {
        "resultType": "input_required",
        "inputRequests": dict(input_requests),
    }
    if request_state is not None:
        result["requestState"] = request_state
    return result


def cacheable(
    result: dict[str, Any], *, ttl_ms: int, cache_scope: str = "public"
) -> dict[str, Any]:
    """Attach cache hints, which list-shaped results are required to carry.

    ``ttlMs`` is a freshness hint that lets a client stop re-listing on every
    turn. ``cacheScope`` decides whether a shared intermediary may hold the
    response: anything whose content depends on the caller's authorization is
    ``private``, because a cache that ignored that would serve one caller's tool
    list to another.
    """
    if ttl_ms < 0:
        raise ValueError("ttl_ms must not be negative")
    if cache_scope not in ("public", "private"):
        raise ValueError('cache_scope must be "public" or "private"')
    return {**result, "ttlMs": ttl_ms, "cacheScope": cache_scope}


def error_response(ident: str | int | None, error: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-RPC error response.

    ``id`` is omitted when the request was malformed enough that no id could be
    read from it.
    """
    out: dict[str, Any] = {"jsonrpc": "2.0"}
    if ident is not None:
        out["id"] = ident
    out["error"] = error
    return out


def result_response(ident: str | int, result: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-RPC result response."""
    return {"jsonrpc": "2.0", "id": ident, "result": result}
