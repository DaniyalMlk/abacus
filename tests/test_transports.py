"""The two transports, and the header validation the HTTP one is required to do."""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import pytest

from abacus.cli import build_server
from abacus.errors import (
    HEADER_MISMATCH,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    UNSUPPORTED_PROTOCOL_VERSION,
)
from abacus.protocol import PROTOCOL_VERSION
from abacus.server import Server
from abacus.transports import StreamableHttp, serve_stdio
from abacus.transports.http import decode_header_value

from .conftest import request as build_request

PRICE_ARGS = {"spot": 100.0, "strike": 100.0, "time": 1.0, "rate": 0.05, "vol": 0.2,
              "type": "call"}


@pytest.fixture
def mcp() -> Server:
    return build_server()


@pytest.fixture
def http(mcp: Server) -> StreamableHttp:
    return StreamableHttp(mcp)


def headers_for(payload: dict[str, Any], **overrides: str) -> dict[str, str]:
    """The headers a conforming client would attach to ``payload``."""
    method = payload["method"]
    out = {
        "Content-Type": "application/json",
        "MCP-Protocol-Version": PROTOCOL_VERSION,
        "Mcp-Method": method,
    }
    params = payload.get("params", {})
    if method in ("tools/call", "prompts/get") and "name" in params:
        out["Mcp-Name"] = params["name"]
    if method == "resources/read" and "uri" in params:
        out["Mcp-Name"] = params["uri"]
    out.update(overrides)
    return out


def body_of(response: Any) -> dict[str, Any]:
    """The response body, asserted present so the tests can index it freely."""
    assert response.body is not None, f"expected a body, got status {response.status}"
    body: dict[str, Any] = response.body
    return body


def post(
    http: StreamableHttp, payload: dict[str, Any], headers: dict[str, str] | None = None
) -> Any:
    return http.handle(
        "POST",
        "/mcp",
        headers if headers is not None else headers_for(payload),
        json.dumps(payload).encode(),
    )


def call_payload(**overrides: Any) -> dict[str, Any]:
    return build_request(
        "tools/call", {"name": "price_european_option", "arguments": {**PRICE_ARGS, **overrides}}
    )


# -- stdio ----------------------------------------------------------------


def test_stdio_answers_each_request_on_its_own_line(mcp: Server) -> None:
    lines = [
        json.dumps(build_request("server/discover", ident=1)),
        json.dumps(call_payload()),
    ]
    out = io.StringIO()
    serve_stdio(mcp, stdin=io.StringIO("\n".join(lines) + "\n"), stdout=out)
    responses = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r["id"] for r in responses] == [1, 1]
    assert responses[0]["result"]["supportedVersions"] == [PROTOCOL_VERSION]


def test_stdio_writes_nothing_for_a_notification(mcp: Server) -> None:
    # Answering one would desynchronise a client counting responses.
    out = io.StringIO()
    serve_stdio(
        mcp,
        stdin=io.StringIO('{"jsonrpc":"2.0","method":"notifications/cancelled"}\n'),
        stdout=out,
    )
    assert out.getvalue() == ""


def test_stdio_skips_blank_lines(mcp: Server) -> None:
    out = io.StringIO()
    serve_stdio(
        mcp,
        stdin=io.StringIO("\n\n" + json.dumps(build_request("server/discover")) + "\n\n"),
        stdout=out,
    )
    assert len(out.getvalue().splitlines()) == 1


def test_stdio_reports_malformed_json_without_stopping(mcp: Server) -> None:
    out = io.StringIO()
    serve_stdio(
        mcp,
        stdin=io.StringIO("{not json\n" + json.dumps(build_request("server/discover")) + "\n"),
        stdout=out,
    )
    responses = [json.loads(line) for line in out.getvalue().splitlines()]
    assert responses[0]["error"]["code"] == -32700
    # The stream survives it, which is the point: one bad line must not end the
    # process a client is depending on.
    assert "result" in responses[1]


# -- HTTP: the happy path -------------------------------------------------


def test_a_conforming_call_succeeds(http: StreamableHttp) -> None:
    response = post(http, call_payload())
    assert response.status == 200
    assert response.body["result"]["structuredContent"]["price"] == pytest.approx(
        10.4505835721, rel=1e-9
    )


def test_a_notification_is_accepted_with_no_body(http: StreamableHttp) -> None:
    payload = {"jsonrpc": "2.0", "method": "notifications/cancelled"}
    response = http.handle(
        "POST",
        "/mcp",
        {"MCP-Protocol-Version": PROTOCOL_VERSION, "Mcp-Method": "notifications/cancelled"},
        json.dumps(payload).encode(),
    )
    assert response.status == 202
    assert response.encoded() == b""


def test_header_names_are_matched_case_insensitively(http: StreamableHttp) -> None:
    payload = call_payload()
    headers = {
        "mcp-protocol-version": PROTOCOL_VERSION,
        "MCP-METHOD": "tools/call",
        "McP-nAmE": "price_european_option",
    }
    assert post(http, payload, headers).status == 200


# -- HTTP: header validation ----------------------------------------------


def test_a_name_header_disagreeing_with_the_body_is_refused(http: StreamableHttp) -> None:
    # The case the check exists for. An intermediary routing on the header would
    # have applied its policy to a different call from the one that ran.
    payload = call_payload()
    response = post(http, payload, headers_for(payload, **{"Mcp-Name": "something_else"}))
    assert response.status == 400
    assert response.body["error"]["code"] == HEADER_MISMATCH
    assert "does not match body value" in response.body["error"]["message"]


def test_a_method_header_disagreeing_with_the_body_is_refused(http: StreamableHttp) -> None:
    payload = call_payload()
    response = post(http, payload, headers_for(payload, **{"Mcp-Method": "tools/list"}))
    assert response.status == 400
    assert response.body["error"]["code"] == HEADER_MISMATCH


def test_a_version_header_disagreeing_with_the_body_is_refused(http: StreamableHttp) -> None:
    payload = call_payload()
    response = post(http, payload, headers_for(payload, **{"MCP-Protocol-Version": "2025-11-25"}))
    assert response.status == 400
    assert response.body["error"]["code"] == HEADER_MISMATCH


@pytest.mark.parametrize("missing", ["MCP-Protocol-Version", "Mcp-Method", "Mcp-Name"])
def test_a_missing_required_header_is_refused(http: StreamableHttp, missing: str) -> None:
    payload = call_payload()
    headers = headers_for(payload)
    del headers[missing]
    response = post(http, payload, headers)
    assert response.status == 400
    assert response.body["error"]["code"] == HEADER_MISMATCH
    assert missing.lower() in response.body["error"]["message"].lower()


def test_the_name_header_is_not_required_for_methods_that_have_no_name(
    http: StreamableHttp,
) -> None:
    payload = build_request("tools/list")
    assert post(http, payload).status == 200


def test_an_error_response_still_quotes_the_request_id(http: StreamableHttp) -> None:
    payload = call_payload()
    payload["id"] = 42
    response = post(http, payload, headers_for(payload, **{"Mcp-Name": "wrong"}))
    assert response.body["id"] == 42


# -- HTTP: the Base64 sentinel --------------------------------------------


def _sentinel(value: str) -> str:
    return "=?base64?" + base64.b64encode(value.encode()).decode() + "?="


def test_a_plain_value_decodes_to_itself() -> None:
    assert decode_header_value("price_european_option") == "price_european_option"


def test_a_sentinel_value_decodes() -> None:
    assert decode_header_value(_sentinel("Hello, 世界")) == "Hello, 世界"


def test_an_encoded_header_is_decoded_before_it_is_compared(http: StreamableHttp) -> None:
    # A server comparing the raw forms would reject this conforming request.
    payload = call_payload()
    headers = headers_for(payload, **{"Mcp-Name": _sentinel("price_european_option")})
    assert post(http, payload, headers).status == 200


def test_an_encoded_header_that_disagrees_is_still_caught(http: StreamableHttp) -> None:
    # And a server that skipped the check for encoded values would have handed
    # an attacker the way around it.
    payload = call_payload()
    headers = headers_for(payload, **{"Mcp-Name": _sentinel("something_else")})
    assert post(http, payload, headers).status == 400


def test_a_value_marked_encoded_that_does_not_decode_is_refused(
    http: StreamableHttp,
) -> None:
    payload = call_payload()
    headers = headers_for(payload, **{"Mcp-Name": "=?base64?!!!not-base64!!!?="})
    response = post(http, payload, headers)
    assert response.status == 400
    assert response.body["error"]["code"] == HEADER_MISMATCH


# -- HTTP: Mcp-Param-* ----------------------------------------------------


def test_a_param_header_matching_its_argument_is_accepted(http: StreamableHttp) -> None:
    payload = call_payload()
    headers = headers_for(payload, **{"Mcp-Param-Type": "call"})
    assert post(http, payload, headers).status == 200


def test_a_param_header_disagreeing_with_its_argument_is_refused(
    http: StreamableHttp,
) -> None:
    payload = call_payload()
    headers = headers_for(payload, **{"Mcp-Param-Type": "put"})
    response = post(http, payload, headers)
    assert response.status == 400
    assert response.body["error"]["code"] == HEADER_MISMATCH


def test_numbers_are_compared_numerically_not_textually(http: StreamableHttp) -> None:
    # 100 and 100.0 are the same JSON number and a client may render either, so
    # a textual comparison would reject a conforming request.
    payload = call_payload()
    headers = headers_for(payload, **{"Mcp-Param-Spot": "100.0"})
    assert post(http, payload, headers).status == 200


def test_an_unrecognised_param_header_is_ignored(http: StreamableHttp) -> None:
    # An intermediary may add headers the server knows nothing about, and the
    # HTTP semantics require forwarding rather than rejecting them.
    payload = call_payload()
    headers = headers_for(payload, **{"Mcp-Param-Tenant": "acme"})
    assert post(http, payload, headers).status == 200


# -- HTTP: status codes ---------------------------------------------------


def test_an_unsupported_version_is_a_bad_request_listing_what_is_supported(
    http: StreamableHttp,
) -> None:
    payload = build_request("tools/list", version="2025-11-25")
    headers = headers_for(payload, **{"MCP-Protocol-Version": "2025-11-25"})
    response = post(http, payload, headers)
    assert response.status == 400
    assert response.body["error"]["code"] == UNSUPPORTED_PROTOCOL_VERSION
    assert response.body["error"]["data"]["supported"] == [PROTOCOL_VERSION]


def test_an_unknown_method_is_a_not_found_carrying_a_json_rpc_body(
    http: StreamableHttp,
) -> None:
    # The body is what distinguishes a modern server that lacks the method from
    # a legacy endpoint that was never there at all.
    payload = build_request("nope/nope")
    response = post(http, payload)
    assert response.status == 404
    assert body_of(response)["error"]["code"] == METHOD_NOT_FOUND


def test_a_wrong_path_is_a_not_found(http: StreamableHttp) -> None:
    payload = call_payload()
    response = http.handle("POST", "/wrong", headers_for(payload), json.dumps(payload).encode())
    assert response.status == 404


def test_a_malformed_body_is_a_bad_request(http: StreamableHttp) -> None:
    response = http.handle(
        "POST",
        "/mcp",
        {"MCP-Protocol-Version": PROTOCOL_VERSION, "Mcp-Method": "tools/list"},
        b"{not json",
    )
    assert response.status == 400
    assert body_of(response)["error"]["code"] == -32700


def test_a_tool_execution_error_is_still_a_200(http: StreamableHttp) -> None:
    # It is a successful request whose result reports a refusal, not a failed
    # request. Returning 4xx would tell an intermediary the call never ran.
    payload = build_request(
        "tools/call", {"name": "price_european_option", "arguments": {"spot": 100}}
    )
    response = post(http, payload)
    assert response.status == 200
    assert body_of(response)["result"]["isError"] is True


def test_a_missing_meta_block_is_a_bad_request(http: StreamableHttp) -> None:
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    response = http.handle(
        "POST",
        "/mcp",
        {"MCP-Protocol-Version": PROTOCOL_VERSION, "Mcp-Method": "tools/list"},
        json.dumps(payload).encode(),
    )
    assert response.status == 400
    assert body_of(response)["error"]["code"] == INVALID_PARAMS


def test_an_oversized_body_is_refused_without_being_parsed(http: StreamableHttp) -> None:
    payload = call_payload()
    response = http.handle(
        "POST", "/mcp", headers_for(payload), b"x" * (http.max_body_bytes + 1)
    )
    assert response.status == 413


# -- HTTP: the older transport --------------------------------------------


@pytest.mark.parametrize("verb", ["GET", "DELETE"])
def test_the_verbs_of_the_older_transport_are_not_allowed(
    http: StreamableHttp, verb: str
) -> None:
    # GET opened the standalone stream and DELETE terminated a session. Neither
    # exists in this revision.
    response = http.handle(verb, "/mcp", {}, b"")
    assert response.status == 405
    assert response.headers["Allow"] == "POST"


def test_a_session_header_from_an_older_client_is_ignored(http: StreamableHttp) -> None:
    # Ignored rather than echoed: minting or reflecting a session id would
    # suggest a session that does not exist.
    payload = call_payload()
    headers = headers_for(payload, **{"Mcp-Session-Id": "abc123"})
    response = post(http, payload, headers)
    assert response.status == 200
    assert "Mcp-Session-Id" not in response.headers


def test_a_last_event_id_header_is_ignored(http: StreamableHttp) -> None:
    # Streams are not resumable in this revision.
    payload = call_payload()
    assert post(http, payload, headers_for(payload, **{"Last-Event-ID": "7"})).status == 200


# -- HTTP: origin ---------------------------------------------------------


def test_an_origin_is_allowed_when_it_is_on_the_list(mcp: Server) -> None:
    transport = StreamableHttp(mcp, allowed_origins=frozenset({"https://app.example"}))
    payload = call_payload()
    headers = headers_for(payload, Origin="https://app.example")
    assert post(transport, payload, headers).status == 200


def test_an_unlisted_origin_is_refused(mcp: Server) -> None:
    # Without this, a page the user visits can drive a local server through DNS
    # rebinding.
    transport = StreamableHttp(mcp, allowed_origins=frozenset({"https://app.example"}))
    payload = call_payload()
    headers = headers_for(payload, Origin="https://evil.example")
    assert post(transport, payload, headers).status == 403


def test_a_request_without_an_origin_is_not_a_browser_request(mcp: Server) -> None:
    transport = StreamableHttp(mcp, allowed_origins=frozenset({"https://app.example"}))
    assert post(transport, call_payload()).status == 200


# -- the assembled server -------------------------------------------------


def test_the_server_declares_only_what_it_implements() -> None:
    server = build_server()
    # No resources or prompts are offered, so neither is declared. Declaring a
    # capability is a promise to answer the requests that go with it.
    assert set(server.capabilities) == {"tools"}
    assert server.capabilities["tools"]["listChanged"] is False


def test_the_instructions_state_the_units(mcp: Server) -> None:
    # This is the text a model reads before composing its first call, so the
    # conventions that cause silent errors belong in it.
    assert mcp.instructions is not None
    assert "0.2" in mcp.instructions
    assert "year fraction" in mcp.instructions
    assert "log(strike / forward)" in mcp.instructions
