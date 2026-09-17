"""Error codes and exceptions for the MCP wire protocol.

JSON-RPC 2.0 reserves ``-32768``..``-32000`` for protocol-level errors and
leaves ``-32000``..``-32099`` to implementations. Revision ``2026-07-28`` of MCP
partitions that implementation range:

``-32000``..``-32019``
    Legacy. Codes here were allocated by implementations before the policy
    existed. New code must not allocate in this sub-range, and a receiver must
    not assume any particular meaning for a code in it.

``-32020``..``-32099``
    Reserved for the specification. Only codes the specification defines may be
    emitted, and only with the meaning it gives them.

The second rule is enforced at construction time rather than left to review:
:class:`ProtocolError` refuses to carry a code in the reserved sub-range unless
the specification defines it. Getting this wrong is invisible in local testing
and only shows up as a client misreading an error, so it is worth a guard.
"""

from __future__ import annotations

from typing import Any

# Standard JSON-RPC 2.0 codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Codes defined by MCP, allocated from the reserved sub-range.
HEADER_MISMATCH = -32020
MISSING_REQUIRED_CLIENT_CAPABILITY = -32021
UNSUPPORTED_PROTOCOL_VERSION = -32022

#: Bounds of the sub-range the specification reserves for itself.
RESERVED_RANGE = (-32099, -32020)

#: Every code the specification defines inside :data:`RESERVED_RANGE`.
SPEC_DEFINED_CODES = frozenset(
    {
        HEADER_MISMATCH,
        MISSING_REQUIRED_CLIENT_CAPABILITY,
        UNSUPPORTED_PROTOCOL_VERSION,
    }
)

#: Codes allocated by earlier revisions and retired. They are never reused, and
#: an implementation of this revision must not emit them. ``-32002`` was
#: resource-not-found before it moved to ``-32602``; ``-32042`` was URL
#: elicitation required, which existed only in ``2025-11-25``.
RETIRED_CODES = frozenset({-32002, -32042})


def is_reserved(code: int) -> bool:
    """Return whether ``code`` falls in the range reserved for the specification."""
    low, high = RESERVED_RANGE
    return low <= code <= high


class ProtocolError(Exception):
    """An error that becomes a JSON-RPC error response.

    Raising one of these anywhere beneath a dispatcher produces a well-formed
    error response. ``data`` is optional and is carried through untouched.
    """

    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        # Retired codes are checked first. Some of them (-32042) also fall inside
        # the reserved range, and "this was retired" tells the reader more about
        # why the code is unavailable than "this is not one the spec defines".
        if code in RETIRED_CODES:
            raise ValueError(
                f"error code {code} was allocated by an earlier protocol revision and "
                "must not be emitted by an implementation of 2026-07-28"
            )
        if is_reserved(code) and code not in SPEC_DEFINED_CODES:
            raise ValueError(
                f"error code {code} is in the range reserved for the MCP specification "
                f"({RESERVED_RANGE[0]}..{RESERVED_RANGE[1]}) but is not a code it defines"
            )
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        """Render as the ``error`` member of a JSON-RPC error response."""
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return error

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code}, message={self.message!r})"


class ParseError(ProtocolError):
    """The payload was not valid JSON."""

    def __init__(self, message: str = "Parse error", data: Any | None = None) -> None:
        super().__init__(PARSE_ERROR, message, data)


class InvalidRequest(ProtocolError):
    """The payload was valid JSON but not a valid JSON-RPC message."""

    def __init__(self, message: str = "Invalid Request", data: Any | None = None) -> None:
        super().__init__(INVALID_REQUEST, message, data)


class MethodNotFound(ProtocolError):
    """No handler is registered for the requested method."""

    def __init__(self, method: str) -> None:
        super().__init__(METHOD_NOT_FOUND, f"Method not found: {method}", {"method": method})
        self.method = method


class InvalidParams(ProtocolError):
    """The request parameters were missing, malformed or of the wrong shape.

    This is also the code for an unknown resource URI, which moved here from
    ``-32002`` in this revision to line up with JSON-RPC.
    """

    def __init__(self, message: str, data: Any | None = None) -> None:
        super().__init__(INVALID_PARAMS, message, data)


class InternalError(ProtocolError):
    """The server failed in a way the caller cannot do anything about."""

    def __init__(self, message: str = "Internal error", data: Any | None = None) -> None:
        super().__init__(INTERNAL_ERROR, message, data)


class HeaderMismatchError(ProtocolError):
    """An HTTP header disagreed with the request body, or a required one was absent.

    Only meaningful on the Streamable HTTP transport, where selected body fields
    are mirrored into headers so intermediaries can route without parsing the
    body. If a load balancer routes on the header while the server executes on
    the body, a disagreement between the two is a security problem rather than a
    cosmetic one, so the request is refused.
    """

    def __init__(self, message: str, data: Any | None = None) -> None:
        super().__init__(HEADER_MISMATCH, message, data)


class MissingRequiredClientCapabilityError(ProtocolError):
    """Handling the request needed a capability the client did not declare.

    A server may not rely on a capability the client has not declared on the
    request in hand, since there is no session in which one could have been
    declared earlier.
    """

    def __init__(self, required: list[str], message: str | None = None) -> None:
        listed = ", ".join(required)
        super().__init__(
            MISSING_REQUIRED_CLIENT_CAPABILITY,
            message or f"Missing required client capability: {listed}",
            {"requiredCapabilities": required},
        )
        self.required = required


class UnsupportedProtocolVersionError(ProtocolError):
    """The request named a protocol version this server does not implement.

    The supported versions travel in ``data`` so the client can pick one and
    retry rather than having to probe.
    """

    def __init__(self, requested: str, supported: list[str]) -> None:
        super().__init__(
            UNSUPPORTED_PROTOCOL_VERSION,
            "Unsupported protocol version",
            {"supported": supported, "requested": requested},
        )
        self.requested = requested
        self.supported = supported
