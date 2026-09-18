"""A conformance suite that drives a server through the wire format.

Every other test in this project reaches the server through a Python call. That
covers the logic and misses the wire — framing, headers, status codes, the shape
of a response as a client actually receives it — and a server can pass all of it
and still be unusable by a real client.

The checks here are written against requirements of the **specification**, not of
this implementation, and they run through :mod:`abacus.client` rather than
against any server object. Both of those are what make the suite worth having:
it can be pointed at a different server, and what it reports is conformance
rather than agreement with whatever this code happens to do.

Where a requirement cannot be tested from outside — a capability gate on a
method this server does not have, say — the check reports ``skip`` with the
reason. A skip is honest and a silent pass is not, and a suite that quietly
counted untested requirements as satisfied would be worse than no suite.

Each check states the requirement it tests in its own words, so a failure report
says what is wrong rather than only which assertion tripped.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from .client import Client, Exchange, HttpTransport, TransportError
from .errors import (
    HEADER_MISMATCH,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    RETIRED_CODES,
    SPEC_DEFINED_CODES,
    UNSUPPORTED_PROTOCOL_VERSION,
    is_reserved,
)
from .protocol import (
    META_CLIENT_CAPABILITIES,
    META_PROTOCOL_VERSION,
    META_SERVER_INFO,
    PROTOCOL_VERSION,
)

#: Result types this revision defines. Anything else is a server inventing one.
RESULT_TYPES = ("complete", "input_required")

#: Cache scopes a list-shaped result may declare.
CACHE_SCOPES = ("public", "private")


class Failure(Exception):
    """A requirement was tested and not met."""


class Skipped(Exception):
    """A requirement could not be tested against this server, with the reason."""


@dataclass(frozen=True)
class Outcome:
    """What happened to one check."""

    name: str
    requirement: str
    scope: str
    status: str
    detail: str = ""

    def line(self) -> str:
        mark = {"pass": "ok  ", "fail": "FAIL", "skip": "skip"}[self.status]
        text = f"{mark}  {self.name}"
        if self.detail:
            text += f"\n        {self.detail}"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requirement": self.requirement,
            "scope": self.scope,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Check:
    """One requirement and the code that tests it."""

    name: str
    requirement: str
    scope: str
    run: Callable[[Context], None]


@dataclass
class Context:
    """What a check has to work with.

    ``http`` is present only when the suite is driving a Streamable HTTP
    endpoint. A check in the ``http`` scope that finds it absent is skipped
    rather than failed, because a stdio server is not required to have one.
    """

    client: Client
    http: HttpTransport | None = None


@dataclass
class Report:
    """The outcome of a run."""

    outcomes: list[Outcome] = field(default_factory=list)

    @property
    def failures(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.status == "fail"]

    @property
    def skipped(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.status == "skip"]

    @property
    def passed(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.status == "pass"]

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self) -> str:
        lines = [outcome.line() for outcome in self.outcomes]
        lines.append("")
        lines.append(
            f"{len(self.passed)} passed, {len(self.failures)} failed, "
            f"{len(self.skipped)} skipped, {len(self.outcomes)} checks"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "passed": len(self.passed),
            "failed": len(self.failures),
            "skipped": len(self.skipped),
            "checks": [outcome.to_dict() for outcome in self.outcomes],
        }


# -- assertions the checks are written in ---------------------------------


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def expect_result(exchange: Exchange, what: str) -> dict[str, Any]:
    result = exchange.result
    if result is None:
        raise Failure(f"{what} returned an error rather than a result: {exchange.error}")
    return result


def expect_error(exchange: Exchange, what: str) -> dict[str, Any]:
    error = exchange.error
    if error is None:
        raise Failure(f"{what} succeeded, but the request should have been refused")
    return error


def expect_code(exchange: Exchange, code: int, what: str) -> dict[str, Any]:
    error = expect_error(exchange, what)
    expect(
        error.get("code") == code,
        f"{what} should be refused with {code}, got {error.get('code')}: "
        f"{error.get('message')}",
    )
    return error


def check_result_envelope(result: dict[str, Any], what: str) -> None:
    """Every result carries a ``resultType`` and this server's identity."""
    kind = result.get("resultType")
    expect(
        kind in RESULT_TYPES,
        f"{what} returned resultType {kind!r}; this revision defines "
        f"{' and '.join(RESULT_TYPES)}",
    )
    meta = result.get("_meta")
    expect(isinstance(meta, dict), f"{what} returned no _meta, so it did not identify itself")
    assert isinstance(meta, dict)
    info = meta.get(META_SERVER_INFO)
    expect(
        isinstance(info, dict)
        and isinstance(info.get("name"), str)
        and isinstance(info.get("version"), str),
        f"{what} did not carry {META_SERVER_INFO} with a name and a version",
    )


def check_cache_hints(result: dict[str, Any], what: str) -> None:
    """A list-shaped result carries freshness and cacheability hints."""
    ttl = result.get("ttlMs")
    expect(
        isinstance(ttl, int) and not isinstance(ttl, bool) and ttl >= 0,
        f"{what} must carry a non-negative integer ttlMs, got {ttl!r}",
    )
    scope = result.get("cacheScope")
    expect(
        scope in CACHE_SCOPES,
        f"{what} must carry a cacheScope of {' or '.join(CACHE_SCOPES)}, got {scope!r}",
    )


def check_error_code_policy(error: dict[str, Any], what: str) -> None:
    """The error-code allocation rules this revision introduced."""
    code = error.get("code")
    expect(isinstance(code, int), f"{what} returned a non-integer error code {code!r}")
    assert isinstance(code, int)
    expect(
        code not in RETIRED_CODES,
        f"{what} used {code}, which an earlier revision allocated and this one retired",
    )
    expect(
        not is_reserved(code) or code in SPEC_DEFINED_CODES,
        f"{what} used {code}, which is inside the range reserved for the "
        "specification but is not a code the specification defines",
    )
    expect(isinstance(error.get("message"), str), f"{what} returned no error message")


# -- the checks ------------------------------------------------------------


def _discovery_is_implemented(context: Context) -> None:
    result = expect_result(context.client.discover(), "server/discover")
    versions = result.get("supportedVersions")
    expect(
        isinstance(versions, list) and all(isinstance(v, str) for v in versions),
        f"server/discover must list supportedVersions as strings, got {versions!r}",
    )
    assert isinstance(versions, list)
    expect(
        context.client.protocol_version in versions,
        f"server answered a {context.client.protocol_version} request but does not "
        f"list it among {versions}",
    )
    expect(
        isinstance(result.get("capabilities"), dict),
        "server/discover must carry a capabilities object",
    )


def _discovery_carries_cache_hints(context: Context) -> None:
    result = expect_result(context.client.discover(), "server/discover")
    check_cache_hints(result, "server/discover")


def _every_result_is_typed_and_attributed(context: Context) -> None:
    for exchange, what in _successful_exchanges(context.client):
        check_result_envelope(expect_result(exchange, what), what)


def _tool_listing_carries_cache_hints(context: Context) -> None:
    result = expect_result(context.client.list_tools(), "tools/list")
    check_cache_hints(result, "tools/list")


def _tool_listing_is_ordered_deterministically(context: Context) -> None:
    first = context.client.tool_names()
    second = context.client.tool_names()
    expect(
        first == second,
        "tools/list must return a deterministic order, so a client can cache a "
        f"listing; got {first} then {second}",
    )
    expect(bool(first), "tools/list returned no tools, so nothing else can be checked")


def _tool_listing_paginates(context: Context) -> None:
    result = expect_result(context.client.list_tools(), "tools/list")
    cursor = result.get("nextCursor")
    if cursor is None:
        raise Skipped("the server returns its whole tool list in one page")
    expect(isinstance(cursor, str), f"nextCursor must be an opaque string, got {cursor!r}")
    assert isinstance(cursor, str)
    following = expect_result(context.client.list_tools(cursor), "tools/list with a cursor")
    expect("tools" in following, "a paged tools/list must still carry tools")


def _an_invalid_cursor_is_refused(context: Context) -> None:
    exchange = context.client.list_tools("this is not a cursor this server issued")
    error = expect_error(exchange, "tools/list with a bogus cursor")
    check_error_code_policy(error, "tools/list with a bogus cursor")


def _tools_declare_their_input_schema(context: Context) -> None:
    result = expect_result(context.client.list_tools(), "tools/list")
    tools = result.get("tools")
    expect(isinstance(tools, list) and bool(tools), "tools/list returned no tools")
    assert isinstance(tools, list)
    for tool in tools:
        expect(isinstance(tool, dict), f"a tool listing entry is not an object: {tool!r}")
        name = tool.get("name")
        expect(isinstance(name, str) and bool(name), f"a tool has no name: {tool!r}")
        expect(
            isinstance(tool.get("inputSchema"), dict),
            f"tool {name!r} declares no inputSchema, so a caller cannot compose a call",
        )
        expect(
            isinstance(tool.get("description"), str),
            f"tool {name!r} has no description",
        )


def _a_successful_call_carries_both_representations(context: Context) -> None:
    name, arguments = _a_callable_tool(context.client)
    result = expect_result(context.client.call_tool(name, arguments), f"tools/call {name}")
    check_result_envelope(result, f"tools/call {name}")
    expect(
        result.get("isError") is False,
        f"tools/call {name} with valid arguments reported isError {result.get('isError')!r}",
    )
    content = result.get("content")
    expect(
        isinstance(content, list) and bool(content),
        f"tools/call {name} returned no content blocks",
    )
    assert isinstance(content, list)
    expect(
        all(isinstance(block, dict) and "type" in block for block in content),
        f"tools/call {name} returned a content block with no type",
    )


def _a_bad_argument_is_a_result_not_a_protocol_error(context: Context) -> None:
    name, _ = _a_callable_tool(context.client)
    exchange = context.client.call_tool(name, {"definitely_not_a_real_field": 1})
    result = expect_result(
        exchange,
        f"tools/call {name} with invalid arguments — an execution failure is a "
        "result with isError set, not a JSON-RPC error, because the caller can "
        "correct it and retry",
    )
    expect(
        result.get("isError") is True,
        f"tools/call {name} with invalid arguments did not set isError",
    )


def _an_unknown_tool_is_refused(context: Context) -> None:
    exchange = context.client.call_tool("no_such_tool_exists_anywhere", {})
    if exchange.result is not None:
        expect(
            exchange.result.get("isError") is True,
            "tools/call for a tool that does not exist reported success",
        )
        return
    check_error_code_policy(expect_error(exchange, "tools/call for an unknown tool"), "it")


def _an_unknown_method_is_method_not_found(context: Context) -> None:
    exchange = context.client.request("abacus/no-such-method")
    expect_code(exchange, METHOD_NOT_FOUND, "an unknown method")


def _an_unsupported_version_is_refused_with_the_alternatives(context: Context) -> None:
    exchange = context.client.request(
        "server/discover", meta=context.client.meta(version="1999-01-01")
    )
    error = expect_code(
        exchange, UNSUPPORTED_PROTOCOL_VERSION, "a request naming an unimplemented revision"
    )
    data = error.get("data")
    expect(
        isinstance(data, dict) and isinstance(data.get("supported"), list),
        "an UnsupportedProtocolVersion error must carry the supported versions in "
        f"data, so a client can pick one and retry; got {data!r}",
    )


def _the_removed_handshake_is_gone(context: Context) -> None:
    exchange = context.client.request("initialize", {"protocolVersion": PROTOCOL_VERSION})
    expect_code(
        exchange,
        METHOD_NOT_FOUND,
        "initialize, which this revision removed — a server still implementing it "
        "is carrying a session model the revision does not have",
    )


def _protocol_version_in_meta_is_required(context: Context) -> None:
    meta = context.client.meta()
    del meta[META_PROTOCOL_VERSION]
    exchange = context.client.request("server/discover", meta=meta)
    expect_code(exchange, INVALID_PARAMS, f"a request with no _meta.{META_PROTOCOL_VERSION}")


def _client_capabilities_in_meta_are_required(context: Context) -> None:
    meta = context.client.meta()
    del meta[META_CLIENT_CAPABILITIES]
    exchange = context.client.request("server/discover", meta=meta)
    expect_code(exchange, INVALID_PARAMS, f"a request with no _meta.{META_CLIENT_CAPABILITIES}")


def _a_batch_is_refused(context: Context) -> None:
    try:
        exchange = context.client.transport.send(
            [context.client.build("server/discover")], expect_response=True
        )
    except TransportError as exc:  # pragma: no cover - a transport may reject it first
        raise Failure(f"a batch was not answered at all: {exc}") from exc
    raw = exchange.raw
    expect(
        isinstance(raw, dict) and "error" in raw,
        "a JSON-RPC batch must be refused; this revision has no batching, and "
        f"answering the first element silently would be worse. Got {raw!r}",
    )


def _a_notification_is_never_answered(context: Context) -> None:
    exchange = context.client.notify("notifications/cancelled", {"requestId": 1})
    expect(
        exchange.raw is None or exchange.raw == "",
        f"a notification must not be answered; got {exchange.raw!r}",
    )
    # On a stream-framed transport a stray answer is not visible at the point it
    # is written — it is visible when it is read in place of the next response.
    # So the check does not stop at looking: it sends a request afterwards and
    # requires the answer to be the answer to that request.
    following = context.client.request("server/discover", ident="after-a-notification")
    expect(
        isinstance(following.raw, dict)
        and following.raw.get("id") == "after-a-notification",
        "the response after a notification was not the response to the request "
        f"that followed it, so something was written for the notification: "
        f"{following.raw!r}",
    )


def _no_error_uses_a_retired_or_reserved_code(context: Context) -> None:
    for exchange, what in _refused_exchanges(context.client):
        check_error_code_policy(expect_error(exchange, what), what)


def _a_string_request_id_is_echoed(context: Context) -> None:
    exchange = context.client.request("server/discover", ident="a-string-id")
    expect(
        isinstance(exchange.raw, dict) and exchange.raw.get("id") == "a-string-id",
        f"a string request id must be echoed unchanged; got {exchange.raw!r}",
    )


# -- Streamable HTTP -------------------------------------------------------


def _require_http(context: Context) -> HttpTransport:
    if context.http is None:
        raise Skipped("this run is not driving a Streamable HTTP endpoint")
    return context.http


def _get_is_refused(context: Context) -> None:
    http = _require_http(context)
    exchange = http.raw_request("GET")
    expect(
        exchange.status == 405,
        "this revision removed the standalone GET stream, so GET on the endpoint "
        f"must answer 405; got {exchange.status}",
    )


def _delete_is_refused(context: Context) -> None:
    http = _require_http(context)
    exchange = http.raw_request("DELETE")
    expect(
        exchange.status == 405,
        "this revision removed sessions, so there is nothing for DELETE to end and "
        f"it must answer 405; got {exchange.status}",
    )


def _a_missing_method_header_is_refused(context: Context) -> None:
    http = _require_http(context)
    client = context.client
    message = client.build("server/discover")
    headers = http.headers_for(message)
    del headers["Mcp-Method"]
    exchange = http.raw_request(
        "POST", headers=headers, body=json.dumps(message).encode()
    )
    expect(
        exchange.status == 400,
        f"a POST with no Mcp-Method header must be refused with 400; got {exchange.status}",
    )
    expect_code(exchange, HEADER_MISMATCH, "a POST with no Mcp-Method header")


def _a_method_header_disagreeing_with_the_body_is_refused(context: Context) -> None:
    http = _require_http(context)
    message = context.client.build("server/discover")
    headers = http.headers_for(message)
    headers["Mcp-Method"] = "tools/list"
    exchange = http.raw_request("POST", headers=headers, body=json.dumps(message).encode())
    expect(
        exchange.status == 400,
        "a header that disagrees with the body means every control in front of the "
        f"server was applied to a different request; got {exchange.status}",
    )
    expect_code(exchange, HEADER_MISMATCH, "a disagreeing Mcp-Method header")


def _a_name_header_disagreeing_with_the_body_is_refused(context: Context) -> None:
    http = _require_http(context)
    name, arguments = _a_callable_tool(context.client)
    message = context.client.build("tools/call", {"name": name, "arguments": arguments})
    headers = http.headers_for(message)
    headers["Mcp-Name"] = "some_other_tool"
    exchange = http.raw_request("POST", headers=headers, body=json.dumps(message).encode())
    expect_code(exchange, HEADER_MISMATCH, "a disagreeing Mcp-Name header")


def _a_base64_name_header_is_decoded_before_comparison(context: Context) -> None:
    http = _require_http(context)
    name, arguments = _a_callable_tool(context.client)
    message = context.client.build("tools/call", {"name": name, "arguments": arguments})
    headers = http.headers_for(message)
    # The same name, wrapped. A server comparing raw forms would reject this
    # conforming request; one skipping the check for wrapped values would accept
    # a disagreeing one, so both directions matter.
    wrapped = base64.b64encode(name.encode()).decode("ascii")
    headers["Mcp-Name"] = f"=?base64?{wrapped}?="
    exchange = http.raw_request("POST", headers=headers, body=json.dumps(message).encode())
    expect(
        exchange.status == 200,
        "a Base64-wrapped header value that decodes to the body value must be "
        f"accepted; got {exchange.status} {exchange.error}",
    )


def _an_unknown_method_answers_404_with_a_body(context: Context) -> None:
    http = _require_http(context)
    message = context.client.build("abacus/no-such-method")
    headers = http.headers_for(message)
    exchange = http.raw_request("POST", headers=headers, body=json.dumps(message).encode())
    expect(
        exchange.status == 404,
        f"an unknown method must answer 404 on this transport; got {exchange.status}",
    )
    expect(
        isinstance(exchange.raw, dict) and "error" in exchange.raw,
        "the 404 must carry a JSON-RPC error body, which is what lets a client "
        "tell a modern server lacking the method from an endpoint that was never "
        f"there; got {exchange.raw!r}",
    )


def _a_refused_request_does_not_answer_200(context: Context) -> None:
    http = _require_http(context)
    meta = context.client.meta()
    del meta[META_PROTOCOL_VERSION]
    message = context.client.build("server/discover", meta=meta)
    headers = http.headers_for(message)
    exchange = http.raw_request("POST", headers=headers, body=json.dumps(message).encode())
    expect(
        exchange.status == 400,
        "a request refused for being malformed must not answer 200, or an "
        f"intermediary counting statuses sees a success; got {exchange.status}",
    )


def _a_session_header_is_ignored(context: Context) -> None:
    http = _require_http(context)
    message = context.client.build("server/discover")
    headers = http.headers_for(message)
    headers["Mcp-Session-Id"] = "a-session-this-revision-does-not-have"
    exchange = http.raw_request("POST", headers=headers, body=json.dumps(message).encode())
    expect(
        exchange.status == 200,
        "this revision has no sessions, so a stray Mcp-Session-Id must be ignored "
        f"rather than acted on; got {exchange.status}",
    )
    expect(
        "mcp-session-id" not in exchange.headers,
        "a server that echoes Mcp-Session-Id is advertising a session model this "
        "revision removed",
    )


# -- helpers ---------------------------------------------------------------

#: Arguments for tools this suite knows how to call. A server offering none of
#: them makes the call-shaped checks unrunnable, and they skip rather than fail.
_KNOWN_ARGUMENTS: dict[str, dict[str, Any]] = {
    "price_european_option": {
        "spot": 100.0,
        "strike": 100.0,
        "time": 0.5,
        "rate": 0.04,
        "vol": 0.2,
        "type": "call",
    }
}


def _a_callable_tool(client: Client) -> tuple[str, dict[str, Any]]:
    for name in client.tool_names():
        if name in _KNOWN_ARGUMENTS:
            return name, dict(_KNOWN_ARGUMENTS[name])
    raise Skipped(
        "this suite has no known-good arguments for any tool the server offers, "
        "so the call-shaped checks cannot be run against it"
    )


def _successful_exchanges(client: Client) -> Iterator[tuple[Exchange, str]]:
    yield client.discover(), "server/discover"
    yield client.list_tools(), "tools/list"


def _refused_exchanges(client: Client) -> Iterator[tuple[Exchange, str]]:
    """A spread of refusals, so the code policy is checked across several paths."""
    yield client.request("abacus/no-such-method"), "an unknown method"
    yield client.list_tools("not-a-cursor"), "an invalid cursor"
    yield (
        client.request("server/discover", meta=client.meta(version="1999-01-01")),
        "an unimplemented revision",
    )
    broken = client.meta()
    del broken[META_PROTOCOL_VERSION]
    yield client.request("server/discover", meta=broken), "a request with no protocol version"
    yield client.request("tools/call", {"name": 42}), "tools/call with a non-string name"


CHECKS: tuple[Check, ...] = (
    Check(
        "discovery/implemented",
        "server/discover is mandatory and lists the revisions the server implements",
        "core",
        _discovery_is_implemented,
    ),
    Check(
        "discovery/cache-hints",
        "a list-shaped result carries ttlMs and cacheScope",
        "core",
        _discovery_carries_cache_hints,
    ),
    Check(
        "result/typed-and-attributed",
        "every result carries resultType, and a server identifies itself on each "
        "one because there is no handshake in which it could have done so once",
        "core",
        _every_result_is_typed_and_attributed,
    ),
    Check(
        "tools/list-cache-hints",
        "tools/list carries ttlMs and cacheScope",
        "core",
        _tool_listing_carries_cache_hints,
    ),
    Check(
        "tools/list-deterministic",
        "tools/list returns a deterministic order, so a client may cache it",
        "core",
        _tool_listing_is_ordered_deterministically,
    ),
    Check(
        "tools/list-pagination",
        "a nextCursor is opaque and may be presented back to continue the listing",
        "core",
        _tool_listing_paginates,
    ),
    Check(
        "tools/invalid-cursor",
        "a cursor the server did not issue is refused with a permitted error code",
        "core",
        _an_invalid_cursor_is_refused,
    ),
    Check(
        "tools/declared-schemas",
        "every listed tool carries a name, a description and an input schema",
        "core",
        _tools_declare_their_input_schema,
    ),
    Check(
        "tools/call-result-shape",
        "a successful tools/call returns content blocks and does not set isError",
        "core",
        _a_successful_call_carries_both_representations,
    ),
    Check(
        "tools/execution-error-is-a-result",
        "a tool that declines a call answers with isError set, not a JSON-RPC "
        "error, because the caller can correct the arguments and retry",
        "core",
        _a_bad_argument_is_a_result_not_a_protocol_error,
    ),
    Check(
        "tools/unknown-tool-refused",
        "calling a tool that does not exist does not report success",
        "core",
        _an_unknown_tool_is_refused,
    ),
    Check(
        "dispatch/unknown-method",
        "an unrecognised method is -32601",
        "core",
        _an_unknown_method_is_method_not_found,
    ),
    Check(
        "version/unsupported-refused",
        "a request naming an unimplemented revision is -32022, carrying the "
        "supported versions so a client can pick one and retry",
        "core",
        _an_unsupported_version_is_refused_with_the_alternatives,
    ),
    Check(
        "version/handshake-removed",
        "initialize was removed in this revision and must not be implemented",
        "core",
        _the_removed_handshake_is_gone,
    ),
    Check(
        "meta/protocol-version-required",
        f"_meta.{META_PROTOCOL_VERSION} is required on every request",
        "core",
        _protocol_version_in_meta_is_required,
    ),
    Check(
        "meta/client-capabilities-required",
        f"_meta.{META_CLIENT_CAPABILITIES} is required on every request",
        "core",
        _client_capabilities_in_meta_are_required,
    ),
    Check(
        "framing/batch-refused",
        "this revision has no batching, so an array payload is refused",
        "core",
        _a_batch_is_refused,
    ),
    Check(
        "framing/notification-unanswered",
        "a notification never receives a response",
        "core",
        _a_notification_is_never_answered,
    ),
    Check(
        "framing/string-id-echoed",
        "a request id is echoed unchanged, whatever its type",
        "core",
        _a_string_request_id_is_echoed,
    ),
    Check(
        "errors/allocation-policy",
        "no error uses a retired code, or a code inside the range reserved for "
        "the specification that the specification does not define",
        "core",
        _no_error_uses_a_retired_or_reserved_code,
    ),
    Check(
        "http/get-refused",
        "the standalone GET stream was removed, so GET answers 405",
        "http",
        _get_is_refused,
    ),
    Check(
        "http/delete-refused",
        "sessions were removed, so DELETE answers 405",
        "http",
        _delete_is_refused,
    ),
    Check(
        "http/method-header-required",
        "Mcp-Method is required on every POST",
        "http",
        _a_missing_method_header_is_refused,
    ),
    Check(
        "http/method-header-validated",
        "a Mcp-Method header disagreeing with the body is HeaderMismatch and 400",
        "http",
        _a_method_header_disagreeing_with_the_body_is_refused,
    ),
    Check(
        "http/name-header-validated",
        "a Mcp-Name header disagreeing with the body is HeaderMismatch",
        "http",
        _a_name_header_disagreeing_with_the_body_is_refused,
    ),
    Check(
        "http/base64-header-decoded",
        "a header value inside the Base64 sentinel is decoded before it is compared",
        "http",
        _a_base64_name_header_is_decoded_before_comparison,
    ),
    Check(
        "http/unknown-method-404",
        "an unknown method answers 404 with a JSON-RPC error body",
        "http",
        _an_unknown_method_answers_404_with_a_body,
    ),
    Check(
        "http/refusal-status",
        "a malformed request does not answer 200",
        "http",
        _a_refused_request_does_not_answer_200,
    ),
    Check(
        "http/session-header-ignored",
        "Mcp-Session-Id is ignored rather than echoed",
        "http",
        _a_session_header_is_ignored,
    ),
)


def run(
    client: Client,
    *,
    http: HttpTransport | None = None,
    checks: Sequence[Check] = CHECKS,
) -> Report:
    """Run ``checks`` against ``client`` and collect the outcomes.

    A check that raises anything other than :class:`Failure` or
    :class:`Skipped` is recorded as a failure naming the exception. A suite that
    crashed partway through would report fewer problems than it found, which is
    the wrong direction for this to fail in.
    """
    context = Context(client=client, http=http)
    report = Report()
    for check in checks:
        try:
            check.run(context)
        except Skipped as exc:
            status, detail = "skip", str(exc)
        except Failure as exc:
            status, detail = "fail", str(exc)
        except Exception as exc:  # a check that crashes is a check that failed
            status, detail = "fail", f"the check itself raised {type(exc).__name__}: {exc}"
        else:
            status, detail = "pass", ""
        report.outcomes.append(
            Outcome(
                name=check.name,
                requirement=check.requirement,
                scope=check.scope,
                status=status,
                detail=detail,
            )
        )
    return report


__all__ = [
    "CHECKS",
    "Check",
    "Context",
    "Failure",
    "Outcome",
    "Report",
    "Skipped",
    "check_cache_hints",
    "check_error_code_policy",
    "check_result_envelope",
    "run",
]
