"""The tool registry, `tools/list` and `tools/call`.

The split this module is built around is the one the specification draws between
two kinds of failure, because the two have different audiences:

*Protocol errors* are JSON-RPC errors. They say the request was not a thing the
server could act on at all — an unknown tool, a malformed body — and a model is
unlikely to recover from one, because the fault is in the plumbing rather than
in the arguments.

*Execution errors* come back as ordinary results with ``isError`` set. They say
the call arrived intact and the server declined to answer it, and they are
addressed to the model: which field, what was wrong with it, what would be
accepted instead. A model that receives one can repair the call and try again,
which is the whole reason for the distinction.

Putting a schema violation in the first category would be the easy mistake. It
is the second: a volatility given as ``20`` instead of ``0.2`` is a fixable
mistake, and telling the caller so is more useful than refusing the request.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import InvalidParams
from .protocol import Request, cacheable, complete
from .schema import Violation, validate

#: Tools per page of `tools/list` when the caller does not constrain it.
PAGE_SIZE = 50

#: Freshness hint on a tool listing. The set changes only when the server is
#: redeployed, but the hint is kept modest so a client that caches it does not
#: hold a stale list for long after one does change.
LIST_TTL_MS = 300_000

ToolHandler = Callable[[dict[str, Any]], Any]


class ToolExecutionError(Exception):
    """A failure the caller can plausibly correct and retry.

    ``details`` carries machine-readable specifics — usually the list of schema
    violations — so a caller is not left parsing prose to work out what to
    change.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str = "invalid_input",
        details: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.details = details or []

    def to_structured(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"error": {"kind": self.kind, "message": self.message}}
        if self.details:
            payload["error"]["details"] = self.details
        return payload


class DomainError(ToolExecutionError):
    """Arguments that satisfy the schema but do not describe a possible market.

    A schema can require that a volatility be a number and not be negative. It
    cannot say that a call priced below its intrinsic value is unreachable, or
    that an option cannot expire in the past. Those checks live in the tools
    themselves and fail through here, in the same shape as a schema violation,
    so the caller has one error format to deal with rather than two.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        details = [{"path": f"/{field}", "keyword": "domain", "message": message}] if field else []
        super().__init__(message, kind="domain", details=details)
        self.field = field


@dataclass(frozen=True)
class Tool:
    """A registered tool: what it is called, what it takes, and what it returns."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    title: str | None = None
    output_schema: dict[str, Any] | None = None
    annotations: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        """Render as the tool object in a `tools/list` result."""
        out: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        if self.title is not None:
            out["title"] = self.title
        if self.output_schema is not None:
            out["outputSchema"] = self.output_schema
        if self.annotations:
            out["annotations"] = dict(self.annotations)
        return out


def _encode_cursor(index: int) -> str:
    """Encode a pagination position.

    A cursor is opaque to the client by contract. Base64 is not a security
    measure; it makes the opacity obvious so nobody starts doing arithmetic on
    the value and depending on a representation that is free to change.
    """
    return base64.urlsafe_b64encode(f"offset:{index}".encode()).decode()


def _decode_cursor(cursor: str) -> int:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode()).decode()
        label, _, number = raw.partition(":")
        if label != "offset":
            raise ValueError(raw)
        index = int(number)
    except (ValueError, UnicodeDecodeError, binascii.Error) as exc:
        raise InvalidParams(f"invalid cursor: {cursor!r}") from exc
    if index < 0:
        raise InvalidParams(f"invalid cursor: {cursor!r}")
    return index


class ToolRegistry:
    """The set of tools a server exposes.

    Insertion order is preserved and used as listing order. The revision asks
    for a deterministic order so clients can cache a listing and so a tool list
    included in a model's context stays byte-identical between turns, which is
    what lets a prompt cache hit.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def register(
        self,
        name: str,
        *,
        description: str,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any] | None = None,
        title: str | None = None,
        annotations: Mapping[str, Any] | None = None,
    ) -> Callable[[ToolHandler], ToolHandler]:
        """Decorator form of :meth:`add`."""

        def decorate(handler: ToolHandler) -> ToolHandler:
            self.add(
                Tool(
                    name=name,
                    description=description,
                    input_schema=input_schema,
                    handler=handler,
                    title=title,
                    output_schema=output_schema,
                    annotations=dict(annotations or {}),
                )
            )
            return handler

        return decorate

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def page(self, cursor: str | None, size: int = PAGE_SIZE) -> tuple[list[Tool], str | None]:
        """Return one page of tools and the cursor for the next, if any."""
        start = _decode_cursor(cursor) if cursor else 0
        tools = list(self._tools.values())
        window = tools[start : start + size]
        following = start + len(window)
        return window, (_encode_cursor(following) if following < len(tools) else None)

    def validate_arguments(self, tool: Tool, arguments: Any) -> dict[str, Any]:
        """Check arguments against a tool's input schema.

        Raises :class:`ToolExecutionError` carrying every violation, rather than
        the first, so a caller with three problems learns about three problems.
        """
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ToolExecutionError(
                f"arguments must be an object, got {type(arguments).__name__}"
            )
        violations: list[Violation] = validate(arguments, tool.input_schema)
        if violations:
            summary = "; ".join(str(v) for v in violations)
            raise ToolExecutionError(
                f"{tool.name} was called with invalid arguments: {summary}",
                details=[v.to_dict() for v in violations],
            )
        return arguments

    def call(self, name: str, arguments: Any) -> dict[str, Any]:
        """Validate and run a tool, returning a `tools/call` result."""
        tool = self.get(name)
        if tool is None:
            # An unknown tool is a protocol error: the caller is working from a
            # tool list that does not match this server, which is not something
            # it can fix by adjusting arguments.
            raise InvalidParams(f"Unknown tool: {name}", {"tool": name, "known": self.names()})

        try:
            checked = self.validate_arguments(tool, arguments)
            payload = tool.handler(checked)
        except ToolExecutionError as exc:
            return error_result(exc)

        return success_result(payload)


def success_result(payload: Any) -> dict[str, Any]:
    """Build a successful `tools/call` result.

    The payload goes into ``structuredContent``, and the same value is
    serialised into a text block. The duplication is deliberate: a client that
    does not read structured content still gets the answer, and a model reading
    the transcript sees the numbers either way.
    """
    return complete(
        content=[{"type": "text", "text": json.dumps(payload, indent=2, sort_keys=True)}],
        structuredContent=payload,
        isError=False,
    )


def error_result(exc: ToolExecutionError) -> dict[str, Any]:
    """Build a failed `tools/call` result.

    This is a *result*, not a JSON-RPC error. The call was well-formed; the
    server is declining to answer it and saying why in a form the caller can act
    on.
    """
    return complete(
        content=[{"type": "text", "text": exc.message}],
        structuredContent=exc.to_structured(),
        isError=True,
    )


def install(server: Any, registry: ToolRegistry) -> None:
    """Wire a registry's tools into a server as `tools/list` and `tools/call`."""

    def _list(request: Request) -> dict[str, Any]:
        cursor = request.argument("cursor")
        if cursor is not None and not isinstance(cursor, str):
            raise InvalidParams("cursor must be a string")
        tools, following = registry.page(cursor)
        result = complete(tools=[tool.describe() for tool in tools])
        if following is not None:
            result["nextCursor"] = following
        # `public`: this listing is the same for every caller. A server whose
        # tool set varied with the caller's granted scopes would have to say
        # `private` here, or a shared cache could hand one caller's list to
        # another.
        return cacheable(result, ttl_ms=LIST_TTL_MS, cache_scope="public")

    def _call(request: Request) -> dict[str, Any]:
        name = request.argument("name")
        if not isinstance(name, str) or not name:
            raise InvalidParams("params.name is required and must be a string")
        return registry.call(name, request.argument("arguments"))

    server.register("tools/list", _list)
    server.register("tools/call", _call)
