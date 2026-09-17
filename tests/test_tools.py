"""The tool surface: listing, calling, and the two kinds of failure."""

from __future__ import annotations

import json
from typing import Any

import pytest

from abacus import tools
from abacus.analytics import default_registry
from abacus.errors import INVALID_PARAMS
from abacus.protocol import complete
from abacus.server import Server, capabilities
from abacus.tools import DomainError, Tool, ToolExecutionError, ToolRegistry

from .conftest import request


@pytest.fixture
def analytics_server() -> Server:
    srv = Server(
        name="abacus",
        version="0.1.0",
        capabilities=capabilities(tools={"listChanged": False}),
    )
    tools.install(srv, default_registry())
    return srv


def call(server: Server, name: str, arguments: Any, ident: int = 1) -> dict[str, Any]:
    response = server.handle(
        request("tools/call", {"name": name, "arguments": arguments}, ident=ident)
    )
    assert response is not None
    return response


def structured(response: dict[str, Any]) -> Any:
    return response["result"]["structuredContent"]


# -- listing --------------------------------------------------------------


def test_list_returns_every_tool_with_its_schema(analytics_server: Server) -> None:
    response = analytics_server.handle(request("tools/list"))
    assert response is not None
    result = response["result"]
    assert result["resultType"] == "complete"
    names = [tool["name"] for tool in result["tools"]]
    assert "price_european_option" in names
    for tool in result["tools"]:
        assert tool["description"]
        assert tool["inputSchema"]["type"] == "object"


def test_list_carries_the_cache_hints_the_revision_requires(analytics_server: Server) -> None:
    result = analytics_server.handle(request("tools/list"))["result"]  # type: ignore[index]
    assert result["ttlMs"] == tools.LIST_TTL_MS
    # This listing is identical for every caller, so a shared cache may hold it.
    assert result["cacheScope"] == "public"


def test_listing_order_is_stable_across_calls(analytics_server: Server) -> None:
    # Required so a client can cache the listing, and so a tool list placed in a
    # model's context is byte-identical between turns and the prompt cache hits.
    first = analytics_server.handle(request("tools/list"))["result"]["tools"]  # type: ignore[index]
    second = analytics_server.handle(request("tools/list"))["result"]["tools"]  # type: ignore[index]
    assert [t["name"] for t in first] == [t["name"] for t in second]


def test_pagination_walks_every_tool_exactly_once() -> None:
    registry = ToolRegistry()
    for index in range(5):
        registry.add(
            Tool(
                name=f"tool_{index}",
                description="x",
                input_schema={"type": "object"},
                handler=lambda args: {},
            )
        )
    seen: list[str] = []
    cursor: str | None = None
    while True:
        page, cursor = registry.page(cursor, size=2)
        seen.extend(tool.name for tool in page)
        if cursor is None:
            break
    assert seen == [f"tool_{i}" for i in range(5)]


def test_the_last_page_carries_no_cursor(analytics_server: Server) -> None:
    result = analytics_server.handle(request("tools/list"))["result"]  # type: ignore[index]
    assert "nextCursor" not in result


def test_a_corrupt_cursor_is_a_protocol_error(analytics_server: Server) -> None:
    response = analytics_server.handle(request("tools/list", {"cursor": "!!!not-a-cursor"}))
    assert response is not None
    assert response["error"]["code"] == INVALID_PARAMS


def test_a_non_string_cursor_is_refused(analytics_server: Server) -> None:
    response = analytics_server.handle(request("tools/list", {"cursor": 3}))
    assert response is not None
    assert response["error"]["code"] == INVALID_PARAMS


# -- the two kinds of failure --------------------------------------------


def test_an_unknown_tool_is_a_protocol_error(analytics_server: Server) -> None:
    # Not something the caller can fix by adjusting arguments: its tool list
    # does not match this server.
    response = call(analytics_server, "price_asian_option", {})
    assert response["error"]["code"] == INVALID_PARAMS
    assert "price_european_option" in response["error"]["data"]["known"]


def test_bad_arguments_are_an_execution_error_not_a_protocol_error(
    analytics_server: Server,
) -> None:
    # The opposite case: the call arrived intact and the arguments are wrong, so
    # it comes back as a result the model can read and act on.
    response = call(analytics_server, "price_european_option", {"spot": 100})
    assert "error" not in response
    assert response["result"]["isError"] is True
    assert structured(response)["error"]["kind"] == "invalid_input"


def test_an_execution_error_names_every_problem_at_once(analytics_server: Server) -> None:
    response = call(
        analytics_server,
        "price_european_option",
        {"spot": -1, "strike": 100, "time": 1, "rate": 0.05, "vol": 0.2, "type": "swap"},
    )
    details = structured(response)["error"]["details"]
    assert {d["path"] for d in details} == {"/spot", "/type"}


def test_a_misspelled_field_is_reported_with_the_accepted_names(
    analytics_server: Server,
) -> None:
    response = call(
        analytics_server,
        "price_european_option",
        {"spot": 100, "strike": 100, "time": 1, "rate": 0.05, "volatility": 0.2, "type": "call"},
    )
    message = structured(response)["error"]["message"]
    assert "volatility" in message
    assert "vol" in message


def test_missing_tool_name_is_a_protocol_error(analytics_server: Server) -> None:
    response = analytics_server.handle(request("tools/call", {"arguments": {}}))
    assert response is not None
    assert response["error"]["code"] == INVALID_PARAMS


def test_non_object_arguments_are_an_execution_error(analytics_server: Server) -> None:
    response = call(analytics_server, "price_european_option", [1, 2, 3])
    assert response["result"]["isError"] is True


def test_absent_arguments_are_treated_as_empty(analytics_server: Server) -> None:
    response = analytics_server.handle(request("tools/call", {"name": "put_call_parity"}))
    assert response is not None
    # Empty, so every required field is reported missing — an execution error,
    # not a crash.
    assert response["result"]["isError"] is True


# -- successful results ---------------------------------------------------


def test_a_result_carries_structured_and_text_content(analytics_server: Server) -> None:
    response = call(
        analytics_server,
        "price_european_option",
        {"spot": 100, "strike": 100, "time": 1, "rate": 0.05, "vol": 0.2, "type": "call"},
    )
    result = response["result"]
    assert result["isError"] is False
    assert result["content"][0]["type"] == "text"
    # The text block holds the same value, so a client that reads only content
    # still gets the answer.
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]


def test_results_conform_to_the_declared_output_schema(analytics_server: Server) -> None:
    from abacus.schema import validate

    registry = default_registry()
    args = {"spot": 100, "strike": 95, "time": 0.5, "rate": 0.03, "vol": 0.25, "type": "put"}
    for name in ("price_european_option", "european_option_greeks"):
        tool = registry.get(name)
        assert tool is not None and tool.output_schema is not None
        payload = structured(call(analytics_server, name, args))
        assert validate(payload, tool.output_schema) == []


# -- registry mechanics ---------------------------------------------------


def test_a_duplicate_tool_is_refused() -> None:
    registry = ToolRegistry()
    tool = Tool(
        name="x", description="d", input_schema={"type": "object"}, handler=lambda args: {}
    )
    registry.add(tool)
    with pytest.raises(ValueError, match="already registered"):
        registry.add(tool)


def test_describe_omits_what_was_not_given() -> None:
    described = Tool(
        name="x", description="d", input_schema={"type": "object"}, handler=lambda args: {}
    ).describe()
    assert set(described) == {"name", "description", "inputSchema"}


def test_describe_includes_title_annotations_and_output_schema() -> None:
    described = Tool(
        name="x",
        description="d",
        input_schema={"type": "object"},
        handler=lambda args: {},
        title="X",
        output_schema={"type": "object"},
        annotations={"readOnlyHint": True},
    ).describe()
    assert described["title"] == "X"
    assert described["annotations"] == {"readOnlyHint": True}
    assert described["outputSchema"] == {"type": "object"}


def test_every_tool_declares_it_is_read_only() -> None:
    # Everything here computes and returns; nothing writes. A client showing a
    # confirmation prompt should be able to tell.
    for tool in default_registry():
        assert tool.annotations["readOnlyHint"] is True


def test_a_tool_raising_an_execution_error_produces_an_error_result() -> None:
    registry = ToolRegistry()

    def _refuse(args: dict[str, Any]) -> dict[str, Any]:
        raise ToolExecutionError("no", kind="teapot", details=[{"path": "/a"}])

    registry.add(
        Tool(name="x", description="d", input_schema={"type": "object"}, handler=_refuse)
    )
    result = registry.call("x", {})
    assert result["isError"] is True
    assert result["structuredContent"]["error"]["kind"] == "teapot"


def test_a_domain_error_points_at_the_offending_field() -> None:
    error = DomainError("bad", field="vol")
    assert error.to_structured()["error"]["details"][0]["path"] == "/vol"
    assert error.to_structured()["error"]["kind"] == "domain"


def test_install_registers_both_methods() -> None:
    srv = Server(name="s", version="1")
    tools.install(srv, ToolRegistry())
    assert srv.implements("tools/list")
    assert srv.implements("tools/call")


def test_results_are_stamped_with_server_identity(analytics_server: Server) -> None:
    from abacus.protocol import META_SERVER_INFO

    response = analytics_server.handle(request("tools/list"))
    assert response is not None
    assert response["result"]["_meta"][META_SERVER_INFO]["name"] == "abacus"


def test_a_handler_returning_a_plain_value_still_works() -> None:
    registry = ToolRegistry()
    registry.add(
        Tool(
            name="x",
            description="d",
            input_schema={"type": "object"},
            handler=lambda args: 42,
        )
    )
    assert registry.call("x", {})["structuredContent"] == 42


def test_success_and_error_results_are_both_marked_complete() -> None:
    assert tools.success_result({})["resultType"] == "complete"
    # An execution error is still a completed request; `isError` is what says
    # the tool declined, not the result type.
    assert tools.error_result(ToolExecutionError("x"))["resultType"] == "complete"
    assert complete()["resultType"] == "complete"
