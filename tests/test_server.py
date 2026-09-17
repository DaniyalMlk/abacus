"""Dispatch: version checks, capability gating, discovery and error rendering."""

from __future__ import annotations

import json
from typing import Any

import pytest

from abacus.errors import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    MISSING_REQUIRED_CLIENT_CAPABILITY,
    PARSE_ERROR,
    UNSUPPORTED_PROTOCOL_VERSION,
)
from abacus.protocol import (
    META_CLIENT_CAPABILITIES,
    META_SERVER_INFO,
    PROTOCOL_VERSION,
    complete,
)
from abacus.server import Server, capabilities

from .conftest import request


def _result(response: dict[str, Any] | None) -> dict[str, Any]:
    assert response is not None, "expected a response"
    assert "error" not in response, f"expected a result, got {response.get('error')}"
    result = response["result"]
    assert isinstance(result, dict)
    return result


def _error(response: dict[str, Any] | None) -> dict[str, Any]:
    assert response is not None, "expected a response"
    assert "error" in response, f"expected an error, got {response.get('result')}"
    error = response["error"]
    assert isinstance(error, dict)
    return error


# -- discovery ------------------------------------------------------------


def test_discover_is_registered_without_being_asked_for() -> None:
    # Servers must implement `server/discover`, so it is not optional wiring.
    assert Server(name="s", version="1").implements("server/discover")


def test_discover_reports_versions_capabilities_and_identity(server: Server) -> None:
    result = _result(server.handle(request("server/discover")))
    assert result["resultType"] == "complete"
    assert result["supportedVersions"] == [PROTOCOL_VERSION]
    assert result["capabilities"] == {"tools": {"listChanged": False}}
    assert result["instructions"] == "A server used by the test suite."
    assert result["_meta"][META_SERVER_INFO] == {"name": "test-server", "version": "0.0.1"}


def test_discover_is_cacheable(server: Server) -> None:
    result = _result(server.handle(request("server/discover")))
    assert result["ttlMs"] == 3_600_000
    assert result["cacheScope"] == "public"


def test_discover_omits_instructions_when_there_are_none() -> None:
    srv = Server(name="s", version="1")
    assert "instructions" not in _result(srv.handle(request("server/discover")))


def test_discover_with_an_unsupported_version_still_reveals_what_is_supported(
    server: Server,
) -> None:
    # A client probing for the server's era gets a recognised modern error here
    # rather than a result, and the error carries the same version list the
    # result would have, so the probe succeeds either way.
    error = _error(server.handle(request("server/discover", version="2025-11-25")))
    assert error["code"] == UNSUPPORTED_PROTOCOL_VERSION
    assert error["data"]["supported"] == [PROTOCOL_VERSION]


# -- dispatch -------------------------------------------------------------


def test_a_registered_method_is_reached(server: Server) -> None:
    result = _result(server.handle(request("test/echo", {"value": "hi"})))
    assert result == {
        "resultType": "complete",
        "echoed": "hi",
        "_meta": {META_SERVER_INFO: {"name": "test-server", "version": "0.0.1"}},
    }


def test_every_result_is_stamped_with_server_identity(server: Server) -> None:
    # There is no handshake in which the server could have identified itself
    # once, so it does so on each result.
    for method in ("server/discover", "test/echo"):
        result = _result(server.handle(request(method)))
        assert result["_meta"][META_SERVER_INFO]["name"] == "test-server"


def test_a_handler_may_override_the_stamped_identity() -> None:
    srv = Server(name="s", version="1")
    identity = {"name": "proxied", "version": "9"}
    srv.register("test/custom", lambda req: {**complete(), "_meta": {META_SERVER_INFO: identity}})
    result = _result(srv.handle(request("test/custom")))
    assert result["_meta"][META_SERVER_INFO]["name"] == "proxied"


def test_unknown_method_is_method_not_found(server: Server) -> None:
    error = _error(server.handle(request("tools/nope")))
    assert error["code"] == METHOD_NOT_FOUND
    assert error["data"] == {"method": "tools/nope"}


def test_registering_a_duplicate_method_is_refused(server: Server) -> None:
    with pytest.raises(ValueError, match="already registered"):
        server.register("test/echo", lambda req: complete())


def test_the_id_is_echoed_on_both_results_and_errors(server: Server) -> None:
    assert server.handle(request("test/echo", ident="abc"))["id"] == "abc"  # type: ignore[index]
    assert server.handle(request("tools/nope", ident=99))["id"] == 99  # type: ignore[index]


# -- version negotiation --------------------------------------------------


def test_unsupported_version_is_rejected_before_the_method_is_looked_up(
    server: Server,
) -> None:
    # The version check runs first, so an unknown method on an unsupported
    # version reports the version — the thing the client can actually act on.
    error = _error(server.handle(request("tools/nope", version="1900-01-01")))
    assert error["code"] == UNSUPPORTED_PROTOCOL_VERSION


def test_version_errors_list_supported_versions(server: Server) -> None:
    error = _error(server.handle(request("test/echo", version="2024-11-05")))
    assert error["data"] == {"supported": [PROTOCOL_VERSION], "requested": "2024-11-05"}


# -- capability gating ----------------------------------------------------


def test_a_method_needing_an_undeclared_capability_is_refused(server: Server) -> None:
    error = _error(server.handle(request("test/needs-elicitation")))
    assert error["code"] == MISSING_REQUIRED_CLIENT_CAPABILITY
    assert error["data"] == {"requiredCapabilities": ["elicitation"]}


def test_the_same_method_succeeds_once_the_capability_is_declared(server: Server) -> None:
    result = _result(
        server.handle(
            request("test/needs-elicitation", client_capabilities={"elicitation": {}})
        )
    )
    assert result["resultType"] == "complete"


def test_capability_gating_names_every_missing_capability() -> None:
    srv = Server(name="s", version="1")
    srv.register("test/greedy", lambda req: complete(), requires=("roots", "elicitation"))
    error = _error(
        srv.handle(request("test/greedy", client_capabilities={"roots": {}}))
    )
    assert error["data"] == {"requiredCapabilities": ["elicitation"]}


# -- malformed input ------------------------------------------------------


def test_missing_meta_is_invalid_params(server: Server) -> None:
    payload = {"jsonrpc": "2.0", "id": 5, "method": "test/echo", "params": {}}
    error = _error(server.handle(payload))
    assert error["code"] == INVALID_PARAMS


def test_missing_required_capability_field_is_invalid_params(server: Server) -> None:
    payload = request("test/echo")
    del payload["params"]["_meta"][META_CLIENT_CAPABILITIES]
    assert _error(server.handle(payload))["code"] == INVALID_PARAMS


def test_malformed_message_is_invalid_request(server: Server) -> None:
    assert _error(server.handle({"jsonrpc": "2.0", "id": 1}))["code"] == INVALID_REQUEST


def test_an_id_is_recovered_from_a_message_that_failed_to_parse(server: Server) -> None:
    # The client has a pending call keyed by this id; answering without it would
    # leave that call to time out.
    response = server.handle({"jsonrpc": "2.0", "id": 7, "method": "test/echo", "params": {}})
    assert response is not None and response["id"] == 7


def test_the_id_is_omitted_when_none_can_be_read(server: Server) -> None:
    response = server.handle("not a message")
    assert response is not None and "id" not in response


def test_a_non_scalar_id_is_not_echoed_back(server: Server) -> None:
    response = server.handle({"jsonrpc": "2.0", "id": {"a": 1}, "method": "m"})
    assert response is not None and "id" not in response


def test_invalid_json_is_a_parse_error(server: Server) -> None:
    assert _error(server.handle_json("{not json"))["code"] == PARSE_ERROR


def test_valid_json_through_the_text_entry_point(server: Server) -> None:
    response = server.handle_json(json.dumps(request("test/echo", {"value": 2})))
    assert _result(response)["echoed"] == 2


# -- notifications --------------------------------------------------------


def test_a_notification_gets_no_response(server: Server) -> None:
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/cancelled"}) is None


def test_a_registered_notification_handler_runs(server: Server) -> None:
    seen: list[str] = []
    server.on_notification("notifications/cancelled", lambda note: seen.append(note.method))
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/cancelled"}) is None
    assert seen == ["notifications/cancelled"]


def test_a_notification_needs_no_protocol_metadata(server: Server) -> None:
    # Header and metadata requirements for notifications are not defined by this
    # revision, so demanding `_meta` on one would reject conforming traffic.
    assert server.handle({"jsonrpc": "2.0", "method": "whatever", "params": {}}) is None


# -- handler faults -------------------------------------------------------


def test_a_raising_handler_becomes_an_internal_error() -> None:
    srv = Server(name="s", version="1")

    def _boom(req: Any) -> dict[str, Any]:
        raise RuntimeError("the floor gave way")

    srv.register("test/boom", _boom)
    error = _error(srv.handle(request("test/boom")))
    assert error["code"] == INTERNAL_ERROR
    assert "the floor gave way" in error["message"]


def test_a_handler_returning_the_wrong_type_becomes_an_internal_error() -> None:
    srv = Server(name="s", version="1")
    srv.register("test/wrong", lambda req: "not a dict")  # type: ignore[arg-type,return-value]
    assert _error(srv.handle(request("test/wrong")))["code"] == INTERNAL_ERROR


# -- capability helper ----------------------------------------------------


def test_capabilities_helper_omits_what_is_not_offered() -> None:
    assert capabilities(tools={}) == {"tools": {}}
    assert capabilities() == {}
    assert capabilities(tools={}, extensions={"io.modelcontextprotocol/tasks": {}}) == {
        "tools": {},
        "extensions": {"io.modelcontextprotocol/tasks": {}},
    }
