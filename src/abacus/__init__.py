"""abacus — option and portfolio analytics exposed over the Model Context Protocol."""

from __future__ import annotations

from .errors import (
    HEADER_MISMATCH,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    MISSING_REQUIRED_CLIENT_CAPABILITY,
    PARSE_ERROR,
    UNSUPPORTED_PROTOCOL_VERSION,
    HeaderMismatchError,
    InternalError,
    InvalidParams,
    InvalidRequest,
    MethodNotFound,
    MissingRequiredClientCapabilityError,
    ParseError,
    ProtocolError,
    UnsupportedProtocolVersionError,
)
from .protocol import (
    PROTOCOL_VERSION,
    SUPPORTED_VERSIONS,
    Capabilities,
    Implementation,
    Notification,
    Request,
    RequestMeta,
    cacheable,
    complete,
    input_required,
    parse_message,
)
from .server import Server, capabilities

__version__ = "0.1.0"

__all__ = [
    "HEADER_MISMATCH",
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "METHOD_NOT_FOUND",
    "MISSING_REQUIRED_CLIENT_CAPABILITY",
    "PARSE_ERROR",
    "PROTOCOL_VERSION",
    "SUPPORTED_VERSIONS",
    "UNSUPPORTED_PROTOCOL_VERSION",
    "Capabilities",
    "HeaderMismatchError",
    "Implementation",
    "InternalError",
    "InvalidParams",
    "InvalidRequest",
    "MethodNotFound",
    "MissingRequiredClientCapabilityError",
    "Notification",
    "ParseError",
    "ProtocolError",
    "Request",
    "RequestMeta",
    "Server",
    "UnsupportedProtocolVersionError",
    "cacheable",
    "capabilities",
    "complete",
    "input_required",
    "parse_message",
    "__version__",
]
