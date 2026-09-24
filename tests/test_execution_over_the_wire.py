"""Every execution tool driven through the protocol, not through a Python call.

``test_execution.py`` calls the handlers directly, which checks the arithmetic
and the refusals and proves nothing about the wire. A tool can be correct in
process and unusable over the protocol in several ways nothing else here would
catch: a payload holding a value JSON cannot carry — which this group very
nearly shipped, since a risk-neutral schedule has an infinite half-life — a
schema that rejects the arguments its own description asks for, an error that
escapes as a stack trace instead of a result a model can act on.

So each of the three is called twice over a transport, once with arguments that
should work and once with arguments that should not, and the second case is
checked for being a *result* with ``isError`` set rather than a JSON-RPC error,
because that is the distinction that decides whether a model can repair the call
and try again. One of them runs against a launched subprocess, which is the only
configuration that exercises the framing as well.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterator
from typing import Any

import pytest

from abacus.cli import build_server
from abacus.client import Client, InProcessTransport, StdioTransport

ORDER: dict[str, Any] = {
    "symbol": "ABC",
    "side": "buy",
    "quantity": 1000.0,
    "fills": [
        {"quantity": 600.0, "price": 100.4, "commission": 6.0},
        {"quantity": 300.0, "price": 100.9, "commission": 3.0},
    ],
    "decisionPrice": 99.8,
    "arrivalPrice": 100.0,
    "finalPrice": 101.5,
    "fees": 0.5,
}

PROBLEM: dict[str, Any] = {
    "quantity": 1_000_000.0,
    "horizon": 5.0,
    "periods": 5,
    "volatility": 0.95,
    "impact": {"gamma": 2.5e-7, "eta": 2.5e-6, "epsilon": 0.0625},
}


@pytest.fixture
def client() -> Iterator[Client]:
    with Client(InProcessTransport(build_server())) as connected:
        yield connected


def succeed(client: Client, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call over the wire and insist on a usable success result."""
    exchange = client.call_tool(name, arguments)
    assert exchange.error is None, exchange.error
    result = exchange.result
    assert result is not None
    assert result["isError"] is False, result["structuredContent"]

    structured = result["structuredContent"]
    text = json.loads(result["content"][0]["text"])
    assert text == structured
    assert isinstance(structured, dict)
    return structured


def be_refused(client: Client, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call over the wire and insist the refusal arrives as a recoverable result."""
    exchange = client.call_tool(name, arguments)
    assert exchange.error is None, f"a fixable mistake became a protocol error: {exchange.error}"
    result = exchange.result
    assert result is not None
    assert result["isError"] is True, result["structuredContent"]
    error = result["structuredContent"]["error"]
    assert isinstance(error, dict)
    assert error["message"], "a refusal with no message tells the caller nothing"
    assert result["content"][0]["text"] == error["message"]
    return error


def test_the_execution_tools_are_all_listed(client: Client) -> None:
    names = client.tool_names()
    for name in (
        "decompose_implementation_shortfall",
        "optimal_execution_schedule",
        "execution_cost_frontier",
    ):
        assert name in names


# -- the three tools, each with a good call and a bad one --------------------


def test_the_decomposition_survives_the_wire(client: Client) -> None:
    result = succeed(client, "decompose_implementation_shortfall", ORDER)
    assert result["side"] == "buy"
    assert result["delayBasis"] == "order"
    assert result["unfilledQuantity"] == pytest.approx(100.0)
    assert math.isclose(sum(result["components"].values()), result["total"], abs_tol=1e-6)
    assert result["totalBps"] == pytest.approx(87.124248, abs=1e-5)


def test_overfilled_fills_come_back_as_a_refusal(client: Client) -> None:
    error = be_refused(
        client,
        "decompose_implementation_shortfall",
        {**ORDER, "fills": [{"quantity": 4000.0, "price": 100.4}]},
    )
    assert error["kind"] == "domain"
    assert "4000" in error["message"]


def test_a_fill_with_a_negative_quantity_is_a_schema_refusal(client: Client) -> None:
    """The schema catches it, and the refusal still arrives as a result.

    A schema violation is a fixable mistake like any other, so it must not become
    a JSON-RPC error — that would tell the caller the request was unprocessable
    when the only problem is one number's sign.
    """
    error = be_refused(
        client,
        "decompose_implementation_shortfall",
        {**ORDER, "fills": [{"quantity": -600.0, "price": 100.4}]},
    )
    assert error["kind"] == "invalid_input"
    assert error["details"]


def test_the_schedule_survives_the_wire(client: Client) -> None:
    result = succeed(
        client, "optimal_execution_schedule", {"problem": PROBLEM, "riskAversion": 1e-6}
    )
    trajectory = result["trajectory"]
    assert math.fsum(trajectory["trades"]) == pytest.approx(PROBLEM["quantity"], rel=1e-9)
    assert trajectory["halfLife"] == pytest.approx(1.64724, abs=1e-4)
    assert result["halfLifeSensitivity"]["elasticityToVolatility"] == pytest.approx(
        -0.970379, abs=1e-4
    )


def test_the_infinite_half_life_crosses_the_wire_as_an_absence(client: Client) -> None:
    """The case that would have shipped invalid JSON.

    A risk-neutral schedule's half-life is infinite, and `json.dumps` writes a
    bare `Infinity` for that. It is not JSON, so a strict parser on the far side
    rejects the whole message — every number in the result, not just the one that
    was unbounded. ``succeed`` re-parses the text block, so this test fails on
    the serialisation rather than on the value.
    """
    result = succeed(
        client, "optimal_execution_schedule", {"problem": PROBLEM, "riskAversion": 0.0}
    )
    assert result["trajectory"]["halfLife"] is None
    assert result["halfLifeSensitivity"] is None
    assert "infinite" in result["note"]
    json.dumps(result, allow_nan=False)


def test_a_schedule_with_no_interior_optimum_comes_back_as_a_refusal(client: Client) -> None:
    """Permanent impact outweighing temporary impact has no schedule to return."""
    error = be_refused(
        client,
        "optimal_execution_schedule",
        {
            "problem": {**PROBLEM, "impact": {"gamma": 1e-3, "eta": 1e-9}},
            "riskAversion": 1e-6,
        },
    )
    assert error["kind"] == "domain"
    assert "permanent impact" in error["message"]


def test_the_frontier_survives_the_wire(client: Client) -> None:
    result = succeed(
        client,
        "execution_cost_frontier",
        {"problem": PROBLEM, "riskAversions": [0.0, 1e-7, 1e-6, 1e-5]},
    )
    costs = [point["expectedCost"] for point in result["points"]]
    risks = [point["costStandardDeviation"] for point in result["points"]]
    assert costs == sorted(costs)
    assert risks == sorted(risks, reverse=True)
    # The risk-neutral point again, this time inside a list: its half-life is the
    # same infinity and has to survive the same way.
    assert result["points"][0]["halfLife"] is None
    json.dumps(result, allow_nan=False)


def test_unsorted_risk_aversions_come_back_as_a_refusal(client: Client) -> None:
    error = be_refused(
        client, "execution_cost_frontier", {"problem": PROBLEM, "riskAversions": [1e-5, 1e-8]}
    )
    assert error["kind"] == "domain"
    assert "increasing" in error["message"]


def test_the_tools_work_over_a_real_pipe() -> None:
    """The same calls against a launched subprocess, which exercises the framing.

    In-process transports share objects; this one shares only bytes. A payload
    that is not serialisable, or a number that does not survive the round trip,
    fails here and passes everywhere else.
    """
    command = [sys.executable, "-m", "abacus.cli", "stdio"]
    with StdioTransport(command) as transport, Client(transport) as client:
        assert client.discover().ok

        decomposition = succeed(client, "decompose_implementation_shortfall", ORDER)
        assert decomposition["components"]["delay"] == pytest.approx(200.0)

        neutral = succeed(
            client, "optimal_execution_schedule", {"problem": PROBLEM, "riskAversion": 0.0}
        )
        assert neutral["trajectory"]["trades"] == [pytest.approx(200_000.0)] * 5
        assert neutral["trajectory"]["halfLife"] is None

        error = be_refused(
            client,
            "execution_cost_frontier",
            {"problem": PROBLEM, "riskAversions": [1e-5, 1e-8]},
        )
        assert error["kind"] == "domain"

        # The server is still serving afterwards, which is what makes the refusal
        # a refusal rather than a crash that happened to be reported.
        assert client.discover().ok
