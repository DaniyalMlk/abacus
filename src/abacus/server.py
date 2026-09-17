"""Method registration and dispatch.

The server object knows nothing about transports. It takes a decoded JSON value
and returns a decoded JSON value, which keeps stdio, Streamable HTTP and the
test harness on exactly the same code path. Anything that needs the network is
somewhere else.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .errors import (
    InternalError,
    InvalidRequest,
    MethodNotFound,
    MissingRequiredClientCapabilityError,
    ParseError,
    ProtocolError,
)
from .protocol import (
    META_SERVER_INFO,
    SUPPORTED_VERSIONS,
    Capabilities,
    Implementation,
    Notification,
    Request,
    cacheable,
    complete,
    error_response,
    parse_message,
    result_response,
)

Handler = Callable[[Request], dict[str, Any]]
NotificationHandler = Callable[[Notification], None]

#: How long a client may treat a discovery result as fresh. A server's identity
#: and capability set change only when the server itself is redeployed, so an
#: hour costs nothing and saves a request per conversation.
DISCOVER_TTL_MS = 3_600_000


@dataclass(frozen=True)
class Method:
    """A registered method and the conditions for reaching its handler."""

    name: str
    handler: Handler
    #: Client capabilities that must be declared on the request. A server may not
    #: rely on a capability the caller did not declare, so anything a handler
    #: needs is named here and checked before the handler runs.
    requires: tuple[str, ...] = ()


@dataclass
class Server:
    """An MCP server: a capability map, a method table, and a dispatcher."""

    name: str
    version: str
    instructions: str | None = None
    title: str | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)
    supported_versions: tuple[str, ...] = SUPPORTED_VERSIONS
    _methods: dict[str, Method] = field(default_factory=dict, repr=False)
    _notifications: dict[str, NotificationHandler] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.register("server/discover", self._discover)

    # -- registration ----------------------------------------------------

    def register(
        self, name: str, handler: Handler, *, requires: Iterable[str] = ()
    ) -> None:
        """Register ``handler`` for ``name``."""
        if name in self._methods:
            raise ValueError(f"method already registered: {name}")
        self._methods[name] = Method(name=name, handler=handler, requires=tuple(requires))

    def method(
        self, name: str, *, requires: Iterable[str] = ()
    ) -> Callable[[Handler], Handler]:
        """Decorator form of :meth:`register`."""

        def decorate(handler: Handler) -> Handler:
            self.register(name, handler, requires=requires)
            return handler

        return decorate

    def on_notification(self, name: str, handler: NotificationHandler) -> None:
        """Register a handler for an incoming notification."""
        if name in self._notifications:
            raise ValueError(f"notification already registered: {name}")
        self._notifications[name] = handler

    def implements(self, name: str) -> bool:
        return name in self._methods

    @property
    def identity(self) -> Implementation:
        return Implementation(name=self.name, version=self.version, title=self.title)

    # -- built-in methods -------------------------------------------------

    def _discover(self, request: Request) -> dict[str, Any]:
        """Answer ``server/discover``, which every server must implement.

        A client may call this before anything else to learn what the server
        supports, and a client that also speaks the older handshake-based
        revisions uses it on stdio to tell which era it is talking to. Note that
        an unsupported version still fails here rather than being answered:
        the resulting error carries the same list of supported versions the
        result would have, so a client learns what it needs either way, and a
        modern error is exactly what the era probe is looking for.
        """
        result = complete(
            supportedVersions=list(self.supported_versions),
            capabilities=dict(self.capabilities),
            **({"instructions": self.instructions} if self.instructions else {}),
        )
        return cacheable(result, ttl_ms=DISCOVER_TTL_MS, cache_scope="public")

    # -- dispatch ---------------------------------------------------------

    def dispatch(self, request: Request) -> dict[str, Any]:
        """Run a parsed request through version and capability checks to its handler."""
        request.meta.require_version(self.supported_versions)

        method = self._methods.get(request.method)
        if method is None:
            raise MethodNotFound(request.method)

        declared = request.meta.client_capabilities
        missing = [cap for cap in method.requires if not declared.declares(cap)]
        if missing:
            raise MissingRequiredClientCapabilityError(missing)

        result = method.handler(request)
        if not isinstance(result, dict):  # pragma: no cover - guards handler mistakes
            raise InternalError(f"handler for {request.method} did not return an object")
        return self._stamp(result)

    def _stamp(self, result: dict[str, Any]) -> dict[str, Any]:
        """Add this server's identity to a result's ``_meta``.

        Servers should identify themselves on every result, since there is no
        handshake in which they could have done so once.
        """
        meta = dict(result.get("_meta") or {})
        meta.setdefault(META_SERVER_INFO, self.identity.to_dict())
        return {**result, "_meta": meta}

    def handle(self, payload: object) -> dict[str, Any] | None:
        """Handle one decoded JSON-RPC message.

        Returns the response to send, or ``None`` for a notification, which
        never gets one.
        """
        try:
            message = parse_message(payload)
        except ProtocolError as exc:
            return error_response(_salvage_id(payload), exc.to_dict())

        if isinstance(message, Notification):
            handler = self._notifications.get(message.method)
            if handler is not None:
                handler(message)
            return None

        try:
            return result_response(message.id, self.dispatch(message))
        except ProtocolError as exc:
            return error_response(message.id, exc.to_dict())
        except Exception as exc:  # noqa: BLE001 - a handler fault must not kill the server
            return error_response(message.id, InternalError(str(exc)).to_dict())

    def handle_json(self, text: str) -> dict[str, Any] | None:
        """Handle one message given as a JSON string."""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return error_response(None, ParseError(f"Parse error: {exc.msg}").to_dict())
        return self.handle(payload)


def _salvage_id(payload: object) -> str | int | None:
    """Recover a usable id from a message that failed to parse.

    An error response should quote the id it is answering wherever that is
    possible, so a client can settle the right pending call instead of having to
    guess or time out. It is only omitted when the message was too malformed for
    an id to be read out of it.
    """
    if not isinstance(payload, dict):
        return None
    ident = payload.get("id")
    if isinstance(ident, bool) or not isinstance(ident, str | int):
        return None
    return ident


def capabilities(
    *,
    tools: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
    prompts: dict[str, Any] | None = None,
    extensions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a server capability map, omitting anything not offered.

    Declaring a capability is a promise to answer the corresponding requests, so
    an absent argument means absent from the map rather than present and empty.
    """
    out: dict[str, Any] = {}
    if tools is not None:
        out["tools"] = tools
    if resources is not None:
        out["resources"] = resources
    if prompts is not None:
        out["prompts"] = prompts
    if extensions is not None:
        out["extensions"] = extensions
    return out


__all__ = [
    "Capabilities",
    "Handler",
    "InvalidRequest",
    "Method",
    "Server",
    "capabilities",
]
