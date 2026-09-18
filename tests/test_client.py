"""The reference client and its three transports.

The in-process transport is exercised throughout the conformance tests, so what
is worth testing here is the framing the client insists on, the headers it
derives, and the two transports that involve real I/O — a launched subprocess
and a socket — which nothing else touches.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from typing import Any

import pytest

from abacus.cli import build_server
from abacus.client import (
    Client,
    Exchange,
    HttpTransport,
    InProcessTransport,
    StdioTransport,
    TransportError,
    encode_header_value,
)
from abacus.conformance import run
from abacus.protocol import (
    META_CLIENT_CAPABILITIES,
    META_CLIENT_INFO,
    META_LOG_LEVEL,
    META_PROTOCOL_VERSION,
    PROTOCOL_VERSION,
)
from abacus.transports import serve_http
from abacus.transports.http import decode_header_value

PRICE_ARGUMENTS = {
    "spot": 100.0,
    "strike": 100.0,
    "time": 0.5,
    "rate": 0.04,
    "vol": 0.2,
    "type": "call",
}


@pytest.fixture
def client() -> Client:
    return Client(InProcessTransport(build_server()))


# -- what the client puts on the wire --------------------------------------


def test_every_request_carries_the_meta_this_revision_requires(client: Client) -> None:
    message = client.build("tools/list")
    meta = message["params"]["_meta"]
    assert meta[META_PROTOCOL_VERSION] == PROTOCOL_VERSION
    assert meta[META_CLIENT_CAPABILITIES] == {}
    assert meta[META_CLIENT_INFO]["name"]


def test_a_log_level_is_only_sent_when_asked_for(client: Client) -> None:
    # Logging is per-request in this revision, and silence is the default; a
    # client that always sent a level would opt every request into notifications.
    assert META_LOG_LEVEL not in client.meta()
    assert client.meta(log_level="warning")[META_LOG_LEVEL] == "warning"


def test_request_ids_are_monotonic_and_notifications_have_none(client: Client) -> None:
    assert client.build("tools/list")["id"] == 1
    assert client.build("tools/list")["id"] == 2
    assert "id" not in client.build("notifications/cancelled", notification=True)


def test_an_explicit_id_does_not_disturb_the_counter(client: Client) -> None:
    client.build("tools/list", ident="custom")
    assert client.build("tools/list")["id"] == 1


# -- the framing the client insists on -------------------------------------


class Canned:
    """A server that answers with whatever it was given."""

    def __init__(self, response: Any) -> None:
        self.response = response

    def handle(self, _payload: Any) -> Any:
        return self.response


@pytest.mark.parametrize(
    ("response", "complaint"),
    [
        ("not an object", "not a JSON object"),
        ({"id": 1, "result": {}}, "jsonrpc"),
        ({"jsonrpc": "2.0", "id": 1}, "exactly one"),
        (
            {"jsonrpc": "2.0", "id": 1, "result": {}, "error": {"code": -1, "message": "x"}},
            "exactly one",
        ),
        ({"jsonrpc": "2.0", "id": 99, "result": {}}, "does not match"),
    ],
)
def test_a_response_that_cannot_be_matched_to_its_request_is_a_transport_error(
    response: Any, complaint: str
) -> None:
    client = Client(InProcessTransport(Canned(response)))
    with pytest.raises(TransportError, match=complaint):
        client.request("server/discover")


def test_the_client_does_not_judge_anything_beyond_the_framing() -> None:
    # A result with no resultType is a conformance failure and not a transport
    # one. The client has to return it, or the suite could not report on it.
    client = Client(InProcessTransport(Canned({"jsonrpc": "2.0", "id": 1, "result": {}})))
    assert client.request("server/discover").result == {}


def test_pagination_is_followed_to_the_end() -> None:
    pages = [
        {"tools": [{"name": "one"}], "nextCursor": "page-2"},
        {"tools": [{"name": "two"}]},
    ]

    class Paged:
        def __init__(self) -> None:
            self.calls = 0

        def handle(self, payload: Any) -> Any:
            page = pages[self.calls]
            self.calls += 1
            return {"jsonrpc": "2.0", "id": payload["id"], "result": page}

    assert Client(InProcessTransport(Paged())).tool_names() == ["one", "two"]


def test_a_repeated_cursor_does_not_loop_forever() -> None:
    # A server that keeps handing back the same cursor would otherwise hang the
    # client, and a conformance run that hangs reports nothing at all.
    class Stuck:
        def handle(self, payload: Any) -> Any:
            return {
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {"tools": [{"name": "one"}], "nextCursor": "always-the-same"},
            }

    with pytest.raises(TransportError, match="repeated a cursor"):
        Client(InProcessTransport(Stuck())).tool_names()


def test_a_failed_listing_is_reported_rather_than_returned_empty(client: Client) -> None:
    class Refusing:
        def handle(self, payload: Any) -> Any:
            return {
                "jsonrpc": "2.0",
                "id": payload["id"],
                "error": {"code": -32601, "message": "no"},
            }

    with pytest.raises(TransportError, match="tools/list failed"):
        Client(InProcessTransport(Refusing())).tool_names()


# -- header encoding --------------------------------------------------------


@pytest.mark.parametrize("value", ["price_european_option", "a b c", "1.5", "{}"])
def test_a_plain_value_travels_as_itself(value: str) -> None:
    assert encode_header_value(value) == value


@pytest.mark.parametrize("value", ["café", " leading", "trailing ", "line\nbreak", "日経"])
def test_a_value_that_cannot_travel_plainly_is_wrapped(value: str) -> None:
    encoded = encode_header_value(value)
    assert encoded.startswith("=?base64?")
    # And the server's decoder is the inverse, which is the property that
    # matters: a wrapped header must compare equal to the body value it mirrors.
    assert decode_header_value(encoded) == value


def test_the_routing_headers_mirror_the_body(client: Client) -> None:
    transport = HttpTransport("http://127.0.0.1:1/mcp")
    message = client.build("tools/call", {"name": "price_european_option", "arguments": {}})
    headers = transport.headers_for(message)
    assert headers["Mcp-Method"] == "tools/call"
    assert headers["Mcp-Name"] == "price_european_option"
    assert headers["MCP-Protocol-Version"] == PROTOCOL_VERSION


def test_a_method_without_a_name_field_gets_no_name_header(client: Client) -> None:
    transport = HttpTransport("http://127.0.0.1:1/mcp")
    assert "Mcp-Name" not in transport.headers_for(client.build("tools/list"))


def test_a_payload_that_is_not_an_object_gets_no_method_header() -> None:
    # A batch has no method to mirror. Inventing one would turn the server's
    # answer about the batch into an answer about a header the client made up.
    transport = HttpTransport("http://127.0.0.1:1/mcp")
    headers = transport.headers_for([{"jsonrpc": "2.0"}])
    assert "Mcp-Method" not in headers
    assert headers["MCP-Protocol-Version"] == PROTOCOL_VERSION


# -- stdio, against a real process ------------------------------------------


def test_a_launched_server_answers_over_stdio() -> None:
    command = [sys.executable, "-m", "abacus.cli", "stdio"]
    with StdioTransport(command) as transport, Client(transport) as client:
        assert client.discover().ok
        result = client.call_tool("price_european_option", PRICE_ARGUMENTS).result
        assert result is not None
        assert result["structuredContent"]["price"] == pytest.approx(6.6270780136, abs=1e-8)


def test_a_command_that_exits_immediately_is_reported_not_hung() -> None:
    with StdioTransport([sys.executable, "-c", "raise SystemExit(3)"]) as transport:
        client = Client(transport)
        with pytest.raises(TransportError):
            client.discover()


def test_a_command_that_writes_rubbish_is_reported() -> None:
    script = "import sys; sys.stdin.readline(); print('not json'); sys.stdout.flush()"
    with StdioTransport([sys.executable, "-c", script]) as transport:
        client = Client(transport)
        with pytest.raises(TransportError, match="not JSON"):
            client.discover()


def test_closing_an_unstarted_stdio_transport_is_harmless() -> None:
    StdioTransport([sys.executable, "-c", "pass"]).close()


def test_the_stdio_entry_point_survives_a_closed_pipe() -> None:
    # What happens when a client goes away: the loop ends rather than raising.
    process = subprocess.run(
        [sys.executable, "-m", "abacus.cli", "stdio"],
        input="",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0


# -- Streamable HTTP, against a real socket ---------------------------------


@pytest.fixture
def endpoint() -> Any:
    """A real server on an ephemeral port, torn down after the test."""
    httpd = serve_http(build_server(), host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/mcp"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_the_client_drives_a_real_endpoint(endpoint: str) -> None:
    with Client(HttpTransport(endpoint)) as client:
        assert len(client.tool_names()) > 0
        result = client.call_tool("price_european_option", PRICE_ARGUMENTS).result
        assert result is not None
        assert result["isError"] is False


def test_a_refusal_keeps_its_status_rather_than_raising(endpoint: str) -> None:
    # A 400 carrying a JSON-RPC error body is a normal answer in this protocol,
    # so it comes back as an exchange with the status attached.
    with Client(HttpTransport(endpoint)) as client:
        meta = client.meta()
        del meta[META_PROTOCOL_VERSION]
        exchange = client.request("server/discover", meta=meta)
        assert exchange.status == 400
        assert exchange.code() == -32602


def test_the_whole_conformance_suite_passes_over_http(endpoint: str) -> None:
    # The end-to-end case: every check that needs a socket, against one.
    transport = HttpTransport(endpoint)
    with Client(transport) as client:
        report = run(client, http=transport)
    assert report.failures == [], report.render()
    assert not any(outcome.scope == "http" for outcome in report.skipped)


def test_an_unreachable_endpoint_is_a_transport_error() -> None:
    transport = HttpTransport("http://127.0.0.1:9/mcp", timeout_s=5)
    with pytest.raises(TransportError, match="could not reach"):
        Client(transport).discover()


def test_an_exchange_exposes_the_halves_of_a_response() -> None:
    ok = Exchange(request={}, raw={"jsonrpc": "2.0", "id": 1, "result": {"a": 1}})
    assert ok.ok and ok.result == {"a": 1} and ok.error is None and ok.code() is None
    bad = Exchange(request={}, raw={"jsonrpc": "2.0", "id": 1, "error": {"code": -32601}})
    assert not bad.ok and bad.result is None and bad.code() == -32601
    empty = Exchange(request={}, raw=None)
    assert not empty.ok and empty.result is None and empty.error is None
